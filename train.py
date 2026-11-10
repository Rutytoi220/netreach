"""
NetReach v2 Training Pipeline: Multi-Task Learning with Dynamic ACT Halting.

Architectural Specs:
- Model: ChallengerLatentNetV2 (Strictly 4.5M - 5.5M parameters).
- Multi-Relational Edge Modeling: 6 relation types (AdminTo, MemberOf, HasSession, AllowedToDelegate, OpenPort, SubnetRoute).
- Autonomous Adaptive Halting (ACT): p_t = sigma(W_halt H_t) with cumulative probability accumulation.
- Multi-Task Loss Weighting: 1.0 * bce + 0.1 * hop_loss + 0.01 * ponder_cost.
- AdamW optimizer, cosine annealing learning rate schedule, gradient clipping (1.0).
- Mixed precision with CUDA bfloat16 autocasting.
- Metrics logged to results/train_log_v2.json and best checkpoints saved to results/checkpoints/.
"""

import os
import json
import time
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from data.generator import create_dataloader, RELATION_TYPES
from netreach.model_v2 import ChallengerLatentNetV2
from models.baseline_transformer import BaselineTransformer
from models.baseline_gru import BaselineGRU

console = Console()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[bold green]Using CUDA Device:[/bold green] {device_name}")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]CUDA not available, using CPU.[/bold yellow]")
    return device


def compute_multitask_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion_bce: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes multi-task weighted loss:
    1.0 * bce + 0.1 * hop_loss + 0.01 * ponder_cost
    """
    labels = batch["label"]
    hop_lengths = batch["hop_length"]

    logits = out["logit"]
    pred_hops = out["hop"]
    ponder_cost = out.get("ponder_cost", torch.tensor(0.0, device=labels.device))

    # Binary reachability loss
    loss_bce = criterion_bce(logits, labels)

    # Hop distance regression loss:
    # Reachable targets have exact shortest path H; unreachable targets have distance 25.0
    hop_targets = torch.where(
        labels == 1.0,
        hop_lengths.float(),
        torch.tensor(25.0, device=labels.device),
    )
    loss_hop = F.smooth_l1_loss(pred_hops, hop_targets)

    # Multi-task loss combination: 1.0 * bce + 0.1 * hop_loss + 0.01 * ponder_cost
    total_loss = 1.0 * loss_bce + 0.1 * loss_hop + 0.01 * ponder_cost

    metrics = {
        "loss_bce": loss_bce.item(),
        "loss_hop": loss_hop.item(),
        "ponder_cost": ponder_cost.item() if torch.is_tensor(ponder_cost) else float(ponder_cost),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion_bce: nn.Module,
    device: torch.device,
    use_bf16: bool,
) -> Dict[str, float]:
    """Train for one epoch with mixed precision, multi-task loss, and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_hop = 0.0
    total_ponder = 0.0
    correct = 0
    total = 0
    act_steps_total = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            if isinstance(model, ChallengerLatentNetV2):
                out = model(batch)  # ACT mode
                loss, loss_dict = compute_multitask_loss(out, batch, criterion_bce)
                logits = out["logit"]
                act_steps = out.get("steps_taken", torch.tensor(0.0)).mean().item()
            else:
                logits = model(batch)
                loss = criterion_bce(logits, labels)
                loss_dict = {"total_loss": loss.item(), "loss_bce": loss.item(), "loss_hop": 0.0, "ponder_cost": 0.0}
                act_steps = 0.0

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        batch_size = len(labels)
        total_loss += loss_dict["total_loss"] * batch_size
        total_bce += loss_dict["loss_bce"] * batch_size
        total_hop += loss_dict["loss_hop"] * batch_size
        total_ponder += loss_dict["ponder_cost"] * batch_size
        act_steps_total += act_steps * batch_size

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "loss_bce": total_bce / total,
        "loss_hop": total_hop / total,
        "ponder_cost": total_ponder / total,
        "acc": correct / total,
        "mean_steps": act_steps_total / total,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
    use_bf16: bool,
    steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate accuracy and loss on a validation dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    hop_mae = 0.0
    act_steps_total = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        hop_lengths = batch["hop_length"]

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            if isinstance(model, ChallengerLatentNetV2):
                out = model(batch, steps=steps)
                loss, _ = compute_multitask_loss(out, batch, criterion_bce)
                logits = out["logit"]
                pred_hops = out["hop"]
                act_steps = out.get("steps_taken", torch.tensor(0.0)).mean().item()
                # Compute MAE on reachable samples
                reach_mask = (labels == 1.0)
                if reach_mask.sum() > 0:
                    hop_mae += F.l1_loss(pred_hops[reach_mask], hop_lengths[reach_mask].float(), reduction="sum").item()
            else:
                logits = model(batch)
                loss = criterion_bce(logits, labels)
                act_steps = 0.0

        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        act_steps_total += act_steps * batch_size
        total += batch_size

    reachable_count = (total // 2) if total > 0 else 1
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "hop_mae": hop_mae / reachable_count,
        "mean_steps": act_steps_total / total,
    }


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
    checkpoint_dir: str,
) -> List[Dict[str, Any]]:
    """Train a single model and return history."""
    console.print(f"\n[bold cyan]=== Starting Training: {model_name} ===[/bold cyan]")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold magenta]{model_name} Parameter Count:[/bold magenta] {param_count:,}")

    if isinstance(model, ChallengerLatentNetV2):
        assert (
            4_500_000 <= param_count <= 5_500_000
        ), f"Parameter count {param_count:,} violates [4.5M, 5.5M] constraint!"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion_bce, device, use_bf16
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion_bce, device, use_bf16)
        dt = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        console.print(
            f"[{model_name}] Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['loss_bce']:.3f}, Hop: {train_metrics['loss_hop']:.3f}, Ponder: {train_metrics['ponder_cost']:.3f}) | "
            f"Train Acc: {train_metrics['acc']*100:.1f}% | "
            f"Val Acc: {val_metrics['acc']*100:.1f}% | "
            f"ACT Steps: {train_metrics['mean_steps']:.1f} | "
            f"LR: {curr_lr:.2e} | Time: {dt:.2f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_hop_mae": val_metrics.get("hop_mae", 0.0),
            "act_steps": train_metrics["mean_steps"],
            "lr": curr_lr,
            "time": dt,
        }
        history.append(record)

        # Save best checkpoint
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "param_count": param_count,
                    "architecture": "ChallengerLatentNetV2" if isinstance(model, ChallengerLatentNetV2) else model_name,
                },
                ckpt_path,
            )

    console.print(f"[bold green]✔ Finished training {model_name}. Best Val Acc: {best_val_acc*100:.1f}%[/bold green]")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train NetReach v2 models with ACT and multi-relational attention.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--train_samples", type=int, default=3500, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=700, help="Number of validation samples")
    parser.add_argument("--num_nodes", type=int, default=48, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=96, help="Target edges per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for logs and checkpoints")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 epoch smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.train_samples = 100
        args.val_samples = 40
        console.print("[bold yellow]Running SMOKE TEST mode (1 epoch, 100 samples).[/bold yellow]")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    console.print(
        f"[cyan]Generating multi-relational training dataset (N={args.num_nodes}, M={args.target_edges}, H in [3, 4, 5], 6 Relations)...[/cyan]"
    )
    train_loader = create_dataloader(
        hop_lengths=[3, 4, 5],
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_nodes=args.num_nodes,
        target_edges=args.target_edges,
        seed=args.seed,
    )
    val_loader = create_dataloader(
        hop_lengths=[3, 4, 5],
        num_samples=args.val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_nodes=args.num_nodes,
        target_edges=args.target_edges,
        seed=args.seed + 1000,
    )

    model = ChallengerLatentNetV2(
        num_nodes=64,
        node_embed_dim=128,
        d_model=384,
        hidden_dim=768,
        nhead=4,
        rel_dim=64,
        num_relations=8,
        max_steps=20,
        epsilon=0.01,
        use_compile=True,
    )

    history = train_model(
        model_name="ChallengerLatentNetV2",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        checkpoint_dir=checkpoint_dir,
    )

    log_path = os.path.join(args.output_dir, "train_log_v2.json")
    with open(log_path, "w") as f:
        json.dump({"ChallengerLatentNetV2": history}, f, indent=2)
    console.print(f"\n[bold green]Saved NetReach v2 training logs to {log_path}[/bold green]")


if __name__ == "__main__":
    main()
