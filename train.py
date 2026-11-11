"""
NetReach v4 Training Pipeline: Momentum Wavefront Recurrence, Collision Peak Halting & Multi-Task Pointer Learning.

Architectural Specs:
- Model: ChallengerLatentNetV4 (Strictly 35.0M - 45.0M parameters).
- Scaled Enterprise Topologies: N = 256 nodes, M = 768 directed edges.
- Multi-Relational Edge Modeling: 6 relation types (AdminTo, MemberOf, HasSession, AllowedToDelegate, OpenPort, SubnetRoute).
- Bidirectional Dual-Wavefront Meet-in-the-Middle Recurrence with Latent Momentum Buffers V_t:
    V_{t+1} = beta * V_t + (1 - beta) * Delta H_t
    H_{t+1} = RMSNorm(H_t + V_{t+1})
- Collision Peak Detection & Halting (Condition A: Hit, Condition B: Convergence/Dead End).
- Scale-Invariant Training Curriculum: H in [4, 16], T_train ~ Uniform(4, 12).
- Multi-Task Loss Weighting: 1.0 * bce + 0.1 * hop_loss + 0.5 * pointer_loss + 0.01 * ponder_cost.
- AdamW optimizer, cosine annealing learning rate schedule, gradient clipping (1.0).
- Mixed precision with CUDA bfloat16 autocasting.
- Metrics logged to results/train_log_v4.json and best checkpoints saved to results/checkpoints/.
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
from netreach.model_v4 import ChallengerLatentNetV4

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


def compute_multitask_loss_v4(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion_bce: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes NetReach v4 multi-task weighted loss:
    1.0 * bce + 0.1 * hop_loss + 0.5 * pointer_loss + 0.01 * ponder_cost
    """
    labels = batch["label"]
    hop_lengths = batch["hop_length"]

    logits = out["logit"]
    pred_hops = out["hop"]
    ptr_logits = out.get("pointer_logits")
    ponder_cost = out.get("ponder_cost", torch.tensor(0.0, device=labels.device))

    # 1. Binary reachability loss
    loss_bce = criterion_bce(logits, labels)

    # 2. Hop distance regression loss:
    # Reachable targets have exact shortest path H; unreachable targets have distance 45.0
    hop_targets = torch.where(
        labels == 1.0,
        hop_lengths.float(),
        torch.tensor(45.0, device=labels.device),
    )
    loss_hop = F.smooth_l1_loss(pred_hops, hop_targets)

    # 3. Pointer Decoder Loss (Predict next hop along active path)
    next_hops = batch.get("next_hop")
    if ptr_logits is not None and next_hops is not None:
        B, N, _ = ptr_logits.shape
        valid_targets = (next_hops != -1)
        if valid_targets.sum() > 0:
            loss_pointer = F.cross_entropy(
                ptr_logits.reshape(-1, N),
                next_hops.reshape(-1),
                ignore_index=-1,
            )
        else:
            loss_pointer = torch.tensor(0.0, device=labels.device)
    else:
        loss_pointer = torch.tensor(0.0, device=labels.device)

    # Multi-task loss combination: 1.0 * bce + 0.1 * hop + 0.5 * pointer + 0.01 * ponder
    total_loss = 1.0 * loss_bce + 0.1 * loss_hop + 0.5 * loss_pointer + 0.01 * ponder_cost

    metrics = {
        "loss_bce": loss_bce.item(),
        "loss_hop": loss_hop.item(),
        "loss_pointer": loss_pointer.item(),
        "ponder_cost": ponder_cost.item() if torch.is_tensor(ponder_cost) else float(ponder_cost),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


# Backward compatibility alias
compute_multitask_loss_v3 = compute_multitask_loss_v4



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
    total_pointer = 0.0
    total_ponder = 0.0
    correct = 0
    total = 0
    steps_total = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)

        # Scale-invariant training: dynamically sample T_train ~ Uniform(4, 12)
        step_budget = random.randint(4, 12)

        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch, steps=step_budget)
                loss, metrics = compute_multitask_loss_v4(out, batch, criterion_bce)
        else:
            out = model(batch, steps=step_budget)
            loss, metrics = compute_multitask_loss_v4(out, batch, criterion_bce)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        logits = out["logit"]
        preds = (torch.sigmoid(logits) >= 0.5).float()
        batch_size = labels.size(0)

        correct += (preds == labels).sum().item()
        total += batch_size
        total_loss += metrics["total_loss"] * batch_size
        total_bce += metrics["loss_bce"] * batch_size
        total_hop += metrics["loss_hop"] * batch_size
        total_pointer += metrics["loss_pointer"] * batch_size
        total_ponder += metrics["ponder_cost"] * batch_size
        steps_total += out["steps_taken"].sum().item()

    return {
        "loss": total_loss / total,
        "loss_bce": total_bce / total,
        "loss_hop": total_hop / total,
        "loss_pointer": total_pointer / total,
        "ponder_cost": total_ponder / total,
        "acc": correct / total,
        "mean_steps": steps_total / total,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
    use_bf16: bool,
    steps: Optional[int] = 8,
) -> Dict[str, float]:
    """Validation evaluation under bidirectional unroll mode (T=8 steps to resolve H in [4, 16])."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    hop_mae = 0.0
    steps_total = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        hop_lengths = batch["hop_length"]

        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch, steps=steps)
                loss, _ = compute_multitask_loss_v4(out, batch, criterion_bce)
        else:
            out = model(batch, steps=steps)
            loss, _ = compute_multitask_loss_v4(out, batch, criterion_bce)

        logits = out["logit"]
        preds = (torch.sigmoid(logits) >= 0.5).float()
        batch_size = labels.size(0)

        correct += (preds == labels).sum().item()
        total += batch_size
        total_loss += loss.item() * batch_size
        steps_total += out["steps_taken"].sum().item()

        # Compute MAE on reachable pairs
        reachable_mask = labels == 1.0
        if reachable_mask.sum() > 0:
            pred_hops = out["hop"][reachable_mask]
            true_hops = hop_lengths[reachable_mask].float()
            hop_mae += (pred_hops - true_hops).abs().sum().item()

    reachable_count = (total // 2) if total > 0 else 1
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "hop_mae": hop_mae / reachable_count,
        "mean_steps": steps_total / total,
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
    """Train NetReach v4 model and return history."""
    console.print(f"\n[bold cyan]=== Starting Training: {model_name} ===[/bold cyan]")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold magenta]{model_name} Parameter Count:[/bold magenta] {param_count:,}")

    assert (
        35_000_000 <= param_count <= 45_000_000
    ), f"Parameter count {param_count:,} violates [35.0M, 45.0M] constraint!"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)
    criterion_bce = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pt")
    best_val_acc = -1.0
    start_epoch = 1
    history = []

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_val_acc = ckpt.get("val_acc", -1.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            console.print(f"[bold green]Resuming from {ckpt_path} (Epoch {ckpt.get('epoch', 0)}, Val Acc: {best_val_acc*100:.1f}%)[/bold green]")
        except Exception as e:
            console.print(f"[bold yellow]Could not resume from checkpoint: {e}[/bold yellow]")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion_bce, device, use_bf16
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion_bce, device, use_bf16)
        dt = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        console.print(
            f"[{model_name}] Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['loss_bce']:.3f}, Hop: {train_metrics['loss_hop']:.3f}, Ptr: {train_metrics['loss_pointer']:.3f}) | "
            f"Train Acc: {train_metrics['acc']*100:.1f}% | "
            f"Val Acc: {val_metrics['acc']*100:.1f}% | "
            f"Peak Halting Steps: {val_metrics['mean_steps']:.1f} | "
            f"LR: {curr_lr:.2e} | Time: {dt:.2f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_hop_mae": val_metrics.get("hop_mae", 0.0),
            "val_halting_steps": val_metrics["mean_steps"],
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
                    "architecture": "ChallengerLatentNetV4",
                },
                ckpt_path,
            )

    console.print(f"[bold green]✔ Finished training {model_name}. Best Val Acc: {best_val_acc*100:.1f}%[/bold green]")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train NetReach v4 models with Momentum Wavefront Recurrence and Collision Peak Halting.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default 4 for safe GPU VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--train_samples", type=int, default=1800, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=360, help="Number of validation samples")
    parser.add_argument("--num_nodes", type=int, default=256, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=768, help="Target edges per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for logs and checkpoints")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 epoch smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.train_samples = 32
        args.val_samples = 16
        args.batch_size = 4
        console.print("[bold yellow]Running SMOKE TEST mode (1 epoch, 32 samples).[/bold yellow]")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    # Training distribution: H in [4, 16]
    train_hops = list(range(4, 17))
    val_hops = list(range(4, 17))

    console.print(f"Generating multi-relational training dataset (N={args.num_nodes}, M={args.target_edges}, H in [4..16], 6 Relations)...")
    train_loader = create_dataloader(
        hop_lengths=train_hops,
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        target_edges=args.target_edges,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = create_dataloader(
        hop_lengths=val_hops,
        num_samples=args.val_samples,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        target_edges=args.target_edges,
        shuffle=False,
        seed=args.seed + 1,
    )

    model = ChallengerLatentNetV4(
        num_nodes=args.num_nodes,
        node_embed_dim=256,
        d_model=640,
        hidden_dim=1280,
        nhead=8,
        rel_dim=128,
        num_relations=8,
        max_steps=22,
    )

    history = train_model(
        model_name="ChallengerLatentNetV4",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        checkpoint_dir=checkpoint_dir,
    )

    log_path = os.path.join(args.output_dir, "train_log_v4.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "model": "ChallengerLatentNetV4",
                "param_count": model.count_parameters(),
                "history": history,
            },
            f,
            indent=2,
        )
    console.print(f"\n[bold green]Saved NetReach v4 training logs to {log_path}[/bold green]")


if __name__ == "__main__":
    main()
