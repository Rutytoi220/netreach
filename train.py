"""
Uniform Training Pipeline for Algorithmic Generalization Benchmark.

Upgrades:
- Step-Invariant Training: Dynamically sample T_train ~ Uniform(3, 8) per batch for ChallengerLatentNet.
- Harder Graph Support: N = 48 nodes, target_edges = 96.
- AdamW optimizer with weight decay 1e-4, Cosine Annealing, Gradient Clipping 1.0, BCEWithLogitsLoss.
- Mixed precision with CUDA bfloat16 autocasting.
- Metrics logged to results/train_log.json and best checkpoints saved to results/checkpoints/.
"""

import os
import json
import time
import random
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
from rich.console import Console

from data.generator import create_dataloader
from models.baseline_transformer import BaselineTransformer
from models.baseline_gru import BaselineGRU
from models.challenger_latent_net import ChallengerLatentNet

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


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    use_bf16: bool,
    step_invariant: bool = True,
) -> Dict[str, float]:
    """Train for one epoch with mixed precision, step-invariant sampling, and gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            if isinstance(model, ChallengerLatentNet):
                # Step-invariant training: dynamically sample T_train ~ Uniform(3, 8)
                t_steps = random.randint(3, 8) if step_invariant else 5
                logits = model(batch, steps=t_steps)
            else:
                logits = model(batch)
            loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_bf16: bool,
    steps: int = 5,
) -> Dict[str, float]:
    """Evaluate accuracy and loss on a validation dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            if isinstance(model, ChallengerLatentNet):
                logits = model(batch, steps=steps)
            else:
                logits = model(batch)
            loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
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
    """Train a single model uniformly and return its history."""
    console.print(f"\n[bold cyan]=== Starting Uniform Training: {model_name} ===[/bold cyan]")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold magenta]{model_name} Parameter Count:[/bold magenta] {param_count:,}")
    assert 400_000 <= param_count <= 600_000, f"Parameter count {param_count} violates [400k, 600k] constraint!"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, use_bf16, step_invariant=True
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, use_bf16, steps=5)
        dt = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        console.print(
            f"[{model_name}] Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']*100:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']*100:.1f}% | "
            f"LR: {curr_lr:.2e} | Time: {dt:.2f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
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
                },
                ckpt_path,
            )

    console.print(f"[bold green]✔ Finished training {model_name}. Best Val Acc: {best_val_acc*100:.1f}%[/bold green]")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train algorithmic reachability models.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--train_samples", type=int, default=4000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=800, help="Number of validation samples")
    parser.add_argument("--num_nodes", type=int, default=48, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=96, help="Target edges per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for logs and checkpoints")
    parser.add_argument("--model", type=str, default="all", choices=["all", "challenger", "baselines"], help="Models to train")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 epoch smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.train_samples = 100
        args.val_samples = 40
        console.print("[bold yellow]Running SMOKE TEST mode (1 epoch, 100 samples).[/bold yellow]")

    # Setup directories
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    # Build Training and Validation DataLoaders (Training distribution: H in [3, 4, 5])
    console.print(f"[cyan]Generating training and validation datasets (N={args.num_nodes}, M={args.target_edges}, H in [3, 4, 5])...[/cyan]")
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

    models_to_train = {}
    if args.model in ["all", "baselines"]:
        models_to_train["BaselineTransformer"] = BaselineTransformer(num_nodes=50)
        models_to_train["BaselineGRU"] = BaselineGRU(num_nodes=50)
    if args.model in ["all", "challenger"]:
        models_to_train["ChallengerLatentNet"] = ChallengerLatentNet(num_nodes=50, train_steps=5)

    full_logs = {}
    # Load existing logs if available
    log_path = os.path.join(args.output_dir, "train_log.json")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                full_logs = json.load(f)
        except Exception:
            full_logs = {}

    for name, model in models_to_train.items():
        history = train_model(
            model_name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_bf16=use_bf16,
            checkpoint_dir=checkpoint_dir,
        )
        full_logs[name] = history

    with open(log_path, "w") as f:
        json.dump(full_logs, f, indent=2)
    console.print(f"\n[bold green]Saved training logs to {log_path}[/bold green]")


if __name__ == "__main__":
    main()
