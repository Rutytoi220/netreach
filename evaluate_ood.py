"""
Out-of-Distribution (OOD) Algorithmic Generalization & Recurrence Scaling Evaluation.

Evaluates trained checkpoints across path lengths:
In-Distribution: H in [3, 5]
Out-of-Distribution: H in [8, 12, 16, 20]

For the Upgraded Challenger Latent Net, test-time latent recurrence compute scaling is evaluated
across test-time iteration counts T_test in [5, 10, 15, 20].

Mathematical Diagnostic Logging:
- Computes average off-diagonal pairwise cosine similarity between node latents at each T_test.
- Detects and flags over-smoothing if cosine similarity approaches 1.0 (threshold > 0.85).

Outputs:
- Rich formatted summary table comparing accuracy, recurrence scaling, and cosine similarity.
- JSON results saved to results/ood_eval_results.json.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from data.generator import create_dataloader
from models.baseline_transformer import BaselineTransformer
from models.baseline_gru import BaselineGRU
from models.challenger_latent_net import ChallengerLatentNet

console = Console()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pairwise_cosine_similarity(H: torch.Tensor) -> float:
    """
    Computes average pairwise off-diagonal cosine similarity between node vectors.
    Args:
        H: Latent node representations [B, N, d].
    Returns:
        float: Mean off-diagonal cosine similarity in [-1, 1].
    """
    B, N, d = H.shape
    if N <= 1:
        return 0.0

    # L2 normalize along the feature dimension d
    H_norm = F.normalize(H, p=2, dim=-1)  # [B, N, d]
    sim_matrix = torch.bmm(H_norm, H_norm.transpose(1, 2))  # [B, N, N]

    # Create mask excluding diagonal elements (self-similarity = 1.0)
    eye_mask = torch.eye(N, dtype=torch.bool, device=H.device).unsqueeze(0).expand(B, -1, -1)
    off_diagonal = sim_matrix[~eye_mask]
    return off_diagonal.mean().item()


@torch.no_grad()
def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_bf16: bool,
    steps: int = None,
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate accuracy on a specific dataloader and compute latent cosine similarity if applicable.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    cosine_sim_sum = 0.0
    cosine_sim_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            if isinstance(model, ChallengerLatentNet):
                logits, latents = model(batch, steps=steps, return_latents=True)
                # Compute pairwise cosine similarity for diagnostic logging
                sim = compute_pairwise_cosine_similarity(latents)
                cosine_sim_sum += sim
                cosine_sim_batches += 1
            else:
                logits = model(batch)
            loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    metrics = {
        "acc": correct / total if total > 0 else 0.0,
        "loss": total_loss / total if total > 0 else 0.0,
    }
    avg_cosine_sim = (cosine_sim_sum / cosine_sim_batches) if cosine_sim_batches > 0 else 0.0
    return metrics, avg_cosine_sim


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please run train.py first.")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    console.print(f"[green]Loaded checkpoint from {ckpt_path} (trained val acc: {ckpt.get('val_acc', 0.0)*100:.1f}%)[/green]")


def run_evaluation(
    eval_hops: List[int],
    samples_per_hop: int,
    t_test_values: List[int],
    checkpoint_dir: str,
    output_dir: str,
    num_nodes: int = 48,
    target_edges: int = 96,
    batch_size: int = 64,
    seed: int = 2026,
):
    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    # Initialize models
    transformer = BaselineTransformer(num_nodes=50).to(device)
    gru = BaselineGRU(num_nodes=50).to(device)
    challenger = ChallengerLatentNet(num_nodes=50, train_steps=5).to(device)

    # Load trained checkpoints
    load_checkpoint(transformer, os.path.join(checkpoint_dir, "baselinetransformer_best.pt"), device)
    load_checkpoint(gru, os.path.join(checkpoint_dir, "baselinegru_best.pt"), device)
    load_checkpoint(challenger, os.path.join(checkpoint_dir, "challengerlatentnet_best.pt"), device)

    results_table = []
    summary_data = {
        "eval_hops": eval_hops,
        "t_test_values": t_test_values,
        "hops": {},
    }

    console.print("\n[bold cyan]Starting Harder Distribution Evaluation Across Path Lengths...[/bold cyan]\n")

    for h in eval_hops:
        dist_type = "In-Dist" if h in [3, 4, 5] else "OOD"
        console.print(f"[bold yellow]Evaluating Hop Length H = {h} ({dist_type})...[/bold yellow]")

        # Create balanced test dataloader for this specific hop length
        loader = create_dataloader(
            hop_lengths=[h],
            num_samples=samples_per_hop,
            batch_size=batch_size,
            shuffle=False,
            num_nodes=num_nodes,
            target_edges=target_edges,
            seed=seed + h * 100,
        )

        # 1. Evaluate Baseline Transformer
        tf_metrics, _ = evaluate_model_on_dataloader(transformer, loader, device, use_bf16)

        # 2. Evaluate Baseline GRU
        gru_metrics, _ = evaluate_model_on_dataloader(gru, loader, device, use_bf16)

        # 3. Evaluate Challenger across T_test values with mathematical diagnostic
        challenger_t_results = {}
        cosine_sim_results = {}
        oversmoothing_flags = {}

        for t_test in t_test_values:
            c_metrics, cos_sim = evaluate_model_on_dataloader(challenger, loader, device, use_bf16, steps=t_test)
            challenger_t_results[f"T={t_test}"] = c_metrics["acc"]
            cosine_sim_results[f"T={t_test}"] = cos_sim
            is_oversmoothed = cos_sim >= 0.85
            oversmoothing_flags[f"T={t_test}"] = is_oversmoothed
            if is_oversmoothed:
                console.print(f"  [bold red]⚠ OVER-SMOOTHING WARNING at T={t_test}: Cosine Sim = {cos_sim:.4f}[/bold red]")

        row = {
            "hop": h,
            "dist": dist_type,
            "transformer_acc": tf_metrics["acc"],
            "gru_acc": gru_metrics["acc"],
            "challenger_accs": challenger_t_results,
            "cosine_sims": cosine_sim_results,
            "oversmoothing_warnings": oversmoothing_flags,
        }
        results_table.append(row)
        summary_data["hops"][str(h)] = row

    # Build and print Rich summary table
    table = Table(
        title="Directed Graph Reachability: Harder Benchmark (N=48, H up to 20)",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("H (Hops)", justify="center", style="cyan", no_wrap=True)
    table.add_column("Type", justify="center", style="bold")
    table.add_column("Transformer\n(Fixed Depth)", justify="center", style="green")
    table.add_column("GRU\n(Sequential)", justify="center", style="blue")

    for t_test in t_test_values:
        table.add_column(f"Challenger\n(T={t_test})", justify="center", style="yellow")

    table.add_column("Cosine Sim\n(T=5 → 20)", justify="center", style="magenta")
    table.add_column("Over-Smoothing\nDiagnostic", justify="center", style="bold")

    for row in results_table:
        dist_style = "[green]In-Dist[/green]" if row["dist"] == "In-Dist" else "[red]OOD[/red]"
        tf_str = f"{row['transformer_acc']*100:.1f}%"
        gru_str = f"{row['gru_acc']*100:.1f}%"
        c_strs = [f"{row['challenger_accs'][f'T={t}']*100:.1f}%" for t in t_test_values]

        cos_t5 = row["cosine_sims"].get("T=5", 0.0)
        cos_t20 = row["cosine_sims"].get(f"T={t_test_values[-1]}", 0.0)
        cos_str = f"{cos_t5:.2f} → {cos_t20:.2f}"

        has_warning = any(row["oversmoothing_warnings"].values())
        diag_str = "[bold red]WARNING[/bold red]" if has_warning else "[bold green]HEALTHY ✔[/bold green]"

        table.add_row(
            str(row["hop"]),
            dist_style,
            tf_str,
            gru_str,
            *c_strs,
            cos_str,
            diag_str,
        )

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ood_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    console.print(f"[bold green]Saved detailed evaluation results to {json_path}[/bold green]\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models across harder in-distribution and OOD hop lengths.")
    parser.add_argument("--samples_per_hop", type=int, default=300, help="Number of test samples per hop length")
    parser.add_argument("--num_nodes", type=int, default=48, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=96, help="Number of edges per graph")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="Path to checkpoints")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output results")
    parser.add_argument("--hops", type=int, nargs="+", default=[3, 5, 8, 12, 16, 20], help="Hops to evaluate")
    parser.add_argument("--t_test", type=int, nargs="+", default=[5, 10, 15, 20], help="Test-time steps for Challenger")
    args = parser.parse_args()

    run_evaluation(
        eval_hops=args.hops,
        samples_per_hop=args.samples_per_hop,
        t_test_values=args.t_test,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_nodes=args.num_nodes,
        target_edges=args.target_edges,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
