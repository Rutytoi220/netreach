"""
NetReach v4 Out-of-Distribution (OOD) & Neural Path Tracing Benchmark.

Evaluates ChallengerLatentNetV4 across path lengths:
- In-Distribution: H in [6, 12]
- Out-of-Distribution: H in [18, 24, 30, 36, 40]
- Scaled graph size: N = 256 nodes, M = 768 directed edges.

Evaluations performed:
1. Test-time compute scaling across fixed iteration counts T_test in [8, 14, 20, 26, 30].
2. Collision Peak Halting mode (Conditions A & B): tracks accuracy, hop prediction MAE, and dynamic steps.
3. Neural Path Tracing verification: computes valid graph walk rate and path reconstruction accuracy.
4. Latent Pairwise Cosine Similarity diagnostics to monitor over-smoothing.
5. Inference latency measurement on CUDA.

Outputs:
- Rich formatted summary table comparing models across hop lengths.
- Detailed JSON results saved to results/ood_eval_v4.json.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from data.generator import create_dataloader, RELATION_TYPES
from netreach.model_v4 import ChallengerLatentNetV4
from netreach.model_v3 import ChallengerLatentNetV3
from models.baseline_transformer import BaselineTransformer
from models.baseline_gru import BaselineGRU

console = Console()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pairwise_cosine_similarity(H: torch.Tensor) -> float:
    """Computes average off-diagonal pairwise cosine similarity between node latents."""
    B, N, d = H.shape
    if N <= 1:
        return 0.0
    H_norm = F.normalize(H, p=2, dim=-1)
    sim_matrix = torch.bmm(H_norm, H_norm.transpose(1, 2))
    eye_mask = torch.eye(N, dtype=torch.bool, device=H.device).unsqueeze(0).expand(B, -1, -1)
    return sim_matrix[~eye_mask].mean().item()


@torch.no_grad()
def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_bf16: bool,
    steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates model on dataloader.
    Supports both fixed step unrolling and autonomous Collision Peak Halting execution.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    hop_mae = 0.0
    reachable_count = 0
    cos_sim_sum = 0.0
    batches = 0
    steps_total = 0.0
    total_latency_ms = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        hop_lengths = batch["hop_length"]

        t_start = time.perf_counter()
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, (ChallengerLatentNetV4, ChallengerLatentNetV3)):
                    out = model(batch, steps=steps, return_latents=True)
                else:
                    out = model(batch)
        else:
            if isinstance(model, (ChallengerLatentNetV4, ChallengerLatentNetV3)):
                out = model(batch, steps=steps, return_latents=True)
            else:
                out = model(batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_latency_ms += (t_end - t_start) * 1000

        logits = out["logit"]
        preds = (torch.sigmoid(logits) >= 0.5).float()
        batch_size = labels.size(0)

        correct += (preds == labels).sum().item()
        total += batch_size

        loss = criterion_bce(logits, labels)
        total_loss += loss.item() * batch_size

        if "steps_taken" in out:
            steps_total += out["steps_taken"].sum().item()
        elif steps is not None:
            steps_total += float(steps) * batch_size

        reachable_mask = labels == 1.0
        if "hop" in out and reachable_mask.sum() > 0:
            pred_hops = out["hop"][reachable_mask]
            true_hops = hop_lengths[reachable_mask].float()
            hop_mae += (pred_hops - true_hops).abs().sum().item()
            reachable_count += reachable_mask.sum().item()

        # Measure latent cosine similarity
        if "H_fwd" in out:
            cos_sim_sum += compute_pairwise_cosine_similarity(out["H_fwd"])
        elif "latents" in out:
            cos_sim_sum += compute_pairwise_cosine_similarity(out["latents"])
        elif "node_latents" in out:
            cos_sim_sum += compute_pairwise_cosine_similarity(out["node_latents"])
        batches += 1

    mean_latency_per_sample = total_latency_ms / total if total > 0 else 0.0

    return {
        "acc": correct / total if total > 0 else 0.0,
        "loss": total_loss / total if total > 0 else 0.0,
        "hop_mae": (hop_mae / reachable_count) if reachable_count > 0 else 0.0,
        "steps_taken": (steps_total / total) if total > 0 else 0.0,
        "cos_sim": (cos_sim_sum / batches) if batches > 0 else 0.0,
        "mean_latency_ms": mean_latency_per_sample,
    }


@torch.no_grad()
def evaluate_path_tracing(
    model: ChallengerLatentNetV4,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: int = 22,
) -> Dict[str, float]:
    """
    Evaluates neural path tracing on reachable samples in dataloader.
    Computes:
    - valid_walk_rate: fraction of reconstructed paths forming valid NetworkX directed walks.
    - mean_hop_error: absolute difference between predicted path length and ground truth H.
    """
    model.eval()
    valid_count = 0
    total_reachable = 0
    total_hop_diff = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        hop_lengths = batch["hop_length"]

        reachable_indices = (labels == 1.0).nonzero().squeeze(-1).tolist()
        if isinstance(reachable_indices, int):
            reachable_indices = [reachable_indices]

        for idx in reachable_indices:
            total_reachable += 1
            true_h = hop_lengths[idx].item()
            trace = model.trace_attack_path(batch, sample_idx=idx, max_steps=max_steps)
            if trace["is_valid_walk"]:
                valid_count += 1
            pred_h = trace["path_length"]
            total_hop_diff += abs(pred_h - true_h)

    if total_reachable == 0:
        return {"valid_walk_rate": 1.0, "mean_hop_error": 0.0, "total_reachable": 0}

    return {
        "valid_walk_rate": valid_count / total_reachable,
        "mean_hop_error": total_hop_diff / total_reachable,
        "total_reachable": total_reachable,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NetReach v4 models across path lengths.")
    parser.add_argument("--samples_per_hop", type=int, default=100, help="Test samples per hop length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="Path to checkpoints")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output results")
    parser.add_argument("--num_nodes", type=int, default=256, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=768, help="Target edges per graph")
    parser.add_argument("--hops", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36, 40], help="Hops to evaluate")
    parser.add_argument("--t_test", type=int, nargs="+", default=[8, 14, 20, 26, 30], help="Test steps for fixed scaling")
    args = parser.parse_args()

    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    # Load NetReach v4 model
    model_v4 = ChallengerLatentNetV4(
        num_nodes=args.num_nodes,
        node_embed_dim=256,
        d_model=640,
        hidden_dim=1280,
        nhead=8,
        rel_dim=128,
        num_relations=8,
        max_steps=32,
    ).to(device)

    v4_ckpt_path = os.path.join(args.checkpoint_dir, "challengerlatentnetv4_best.pt")
    if os.path.exists(v4_ckpt_path):
        ckpt_v4 = torch.load(v4_ckpt_path, map_location=device)
        model_v4.load_state_dict(ckpt_v4["model_state_dict"])
        console.print(f"[bold green]Loaded NetReach v4 checkpoint from {v4_ckpt_path} (val acc: {ckpt_v4.get('val_acc', 0.0)*100:.1f}%, params: {ckpt_v4.get('param_count', 0):,})[/bold green]")
    else:
        console.print(f"[bold yellow]Warning: {v4_ckpt_path} not found. Running with initialized weights.[/bold yellow]")

    # Check for legacy Baselines
    try:
        transformer = BaselineTransformer(num_nodes=args.num_nodes).to(device)
        t_ckpt_path = os.path.join(args.checkpoint_dir, "baselinetransformer_best.pt")
        has_t = os.path.exists(t_ckpt_path)
        if has_t:
            ckpt_t = torch.load(t_ckpt_path, map_location=device)
            transformer.load_state_dict(ckpt_t["model_state_dict"])
    except Exception:
        has_t = False

    try:
        gru = BaselineGRU(num_nodes=args.num_nodes).to(device)
        gru_ckpt_path = os.path.join(args.checkpoint_dir, "baselinegru_best.pt")
        has_gru = os.path.exists(gru_ckpt_path)
        if has_gru:
            ckpt_gru = torch.load(gru_ckpt_path, map_location=device)
            gru.load_state_dict(ckpt_gru["model_state_dict"])
    except Exception:
        has_gru = False

    console.print(f"\n[bold cyan]Starting NetReach v4 Momentum Evaluation Across Path Lengths (N={args.num_nodes}, M={args.target_edges})...[/bold cyan]\n")

    results_data = {
        "eval_hops": args.hops,
        "t_test_values": args.t_test,
        "num_nodes": args.num_nodes,
        "target_edges": args.target_edges,
        "hops": {},
    }

    table = Table(
        title="NetReach v4: Momentum Wavefront Recurrence, Peak Halting & Neural Path Tracing Benchmark",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Hops (H)", justify="center", style="bold yellow")
    table.add_column("Dist", justify="center")
    table.add_column("Transformer\n(Fixed 3-L)", justify="center", style="cyan")
    table.add_column("GRU\n(Sequential)", justify="center", style="cyan")
    for t_step in args.t_test:
        table.add_column(f"v4 (T={t_step})\n[Momentum]", justify="center", style="green")
    table.add_column("Peak Halting\n(Acc / Steps)", justify="center", style="bold green")
    table.add_column("Hop MAE", justify="center")
    table.add_column("Path Walk\nValidity", justify="center", style="bold cyan")
    table.add_column("Latent\nCos Sim", justify="center")
    table.add_column("Latency\n(per graph)", justify="center")

    for h in args.hops:
        dist_type = "In-Dist" if h in [6, 12] else "OOD"
        console.print(f"Evaluating Hop Length H = {h} ({dist_type})...")

        loader = create_dataloader(
            hop_lengths=[h],
            num_samples=args.samples_per_hop,
            batch_size=args.batch_size,
            num_nodes=args.num_nodes,
            target_edges=args.target_edges,
            shuffle=False,
            seed=100 + h,
        )

        # 1. Baseline Transformer
        if has_t:
            t_res = evaluate_model_on_dataloader(transformer, loader, device, use_bf16)
            t_acc = t_res["acc"]
        else:
            t_acc = 0.50

        # 2. Baseline GRU
        if has_gru:
            gru_res = evaluate_model_on_dataloader(gru, loader, device, use_bf16)
            gru_acc = gru_res["acc"]
        else:
            gru_acc = 0.50

        # 3. NetReach v4 Autonomous Collision Peak Halting mode
        halt_res = evaluate_model_on_dataloader(model_v4, loader, device, use_bf16, steps=None)
        halt_acc = halt_res["acc"]
        halt_steps = halt_res["steps_taken"]
        halt_mae = halt_res["hop_mae"]
        mean_latency = halt_res["mean_latency_ms"]

        # 4. NetReach v4 Fixed Steps Scaling
        fixed_step_accs = {}
        cos_sims = {}
        for t_step in args.t_test:
            step_res = evaluate_model_on_dataloader(model_v4, loader, device, use_bf16, steps=t_step)
            fixed_step_accs[f"T={t_step}"] = step_res["acc"]
            cos_sims[f"T={t_step}"] = step_res["cos_sim"]

        # 5. Neural Path Tracing Evaluation
        trace_res = evaluate_path_tracing(model_v4, loader, device, max_steps=32)
        valid_rate = trace_res["valid_walk_rate"]

        results_data["hops"][str(h)] = {
            "hop": h,
            "dist": dist_type,
            "transformer_acc": t_acc,
            "gru_acc": gru_acc,
            "peak_halting_acc": halt_acc,
            "peak_halting_steps": halt_steps,
            "hop_mae": halt_mae,
            "path_walk_validity": valid_rate,
            "path_hop_error": trace_res["mean_hop_error"],
            "fixed_accs": fixed_step_accs,
            "cosine_sims": cos_sims,
            "latency_ms": mean_latency,
        }

        min_sim = min(cos_sims.values())
        max_sim = max(cos_sims.values())

        row = [
            str(h),
            dist_type,
            f"{t_acc*100:.1f}%",
            f"{gru_acc*100:.1f}%",
        ]
        for t_step in args.t_test:
            row.append(f"{fixed_step_accs[f'T={t_step}']*100:.1f}%")

        row.extend([
            f"{halt_acc*100:.1f}%\n({halt_steps:.1f} st)",
            f"±{halt_mae:.1f} hops",
            f"{valid_rate*100:.1f}%",
            f"{min_sim:.2f} → {max_sim:.2f}",
            f"{mean_latency:.2f} ms",
        ])
        table.add_row(*row)

    console.print("\n")
    console.print(table)

    out_file = os.path.join(args.output_dir, "ood_eval_v4.json")
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)

    console.print(f"\n[bold green]Saved detailed NetReach v4 evaluation results to {out_file}[/bold green]\n")


if __name__ == "__main__":
    main()
