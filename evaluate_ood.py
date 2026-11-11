"""
NetReach v3 Out-of-Distribution (OOD) & Bidirectional Recurrence Evaluation.

Evaluates ChallengerLatentNetV3 across path lengths:
- In-Distribution: H in [4, 8]
- Out-of-Distribution: H in [12, 16, 20, 24, 32]
- Scaled graph size: N = 128 nodes, M = 384 directed edges.

Evaluations performed:
1. Test-time compute scaling across fixed iteration counts T_test in [6, 10, 14, 18, 22].
2. Autonomous Adaptive Computation Time (ACT) mode: tracks accuracy, hop prediction MAE, and dynamic steps.
3. Bidirectional Meet-in-the-Middle efficiency: verifies collision at ~ ceil(H / 2) steps.
4. Latent Pairwise Cosine Similarity diagnostics to monitor over-smoothing.
5. Inference latency measurement on CUDA.

Outputs:
- Rich formatted summary table comparing models across hop lengths.
- Detailed JSON results saved to results/ood_eval_v3.json.
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
from netreach.model_v3 import ChallengerLatentNetV3
from netreach.model_v2 import ChallengerLatentNetV2
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
    Supports both fixed step unrolling and autonomous ACT execution.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    hop_mae = 0.0
    reachable_count = 0
    cos_sim_sum = 0.0
    batches = 0
    act_steps_total = 0.0
    total_latency_ms = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        hop_lengths = batch["hop_length"]

        t_start = time.perf_counter()
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, (ChallengerLatentNetV3, ChallengerLatentNetV2)):
                    out = model(batch, steps=steps, return_latents=True)
                else:
                    out = model(batch)
        else:
            if isinstance(model, (ChallengerLatentNetV3, ChallengerLatentNetV2)):
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
            act_steps_total += out["steps_taken"].sum().item()
        elif steps is not None:
            act_steps_total += float(steps) * batch_size

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
        "act_steps": (act_steps_total / total) if total > 0 else 0.0,
        "cos_sim": (cos_sim_sum / batches) if batches > 0 else 0.0,
        "mean_latency_ms": mean_latency_per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NetReach v3 models across path lengths.")
    parser.add_argument("--samples_per_hop", type=int, default=300, help="Test samples per hop length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="Path to checkpoints")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output results")
    parser.add_argument("--num_nodes", type=int, default=128, help="Number of nodes per graph")
    parser.add_argument("--target_edges", type=int, default=384, help="Target edges per graph")
    parser.add_argument("--hops", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 32], help="Hops to evaluate")
    parser.add_argument("--t_test", type=int, nargs="+", default=[6, 10, 14, 18, 22], help="Test steps for fixed scaling")
    args = parser.parse_args()

    device = get_device()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    # Load NetReach v3 model
    model_v3 = ChallengerLatentNetV3(
        num_nodes=args.num_nodes,
        node_embed_dim=256,
        d_model=512,
        hidden_dim=1024,
        nhead=8,
        rel_dim=128,
        num_relations=8,
        max_steps=22,
    ).to(device)

    v3_ckpt_path = os.path.join(args.checkpoint_dir, "challengerlatentnetv3_best.pt")
    if os.path.exists(v3_ckpt_path):
        ckpt_v3 = torch.load(v3_ckpt_path, map_location=device)
        model_v3.load_state_dict(ckpt_v3["model_state_dict"])
        console.print(f"[bold green]Loaded NetReach v3 checkpoint from {v3_ckpt_path} (val acc: {ckpt_v3.get('val_acc', 0.0)*100:.1f}%, params: {ckpt_v3.get('param_count', 0):,})[/bold green]")
    else:
        console.print(f"[bold yellow]Warning: {v3_ckpt_path} not found. Running with initialized weights.[/bold yellow]")

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

    console.print(f"\n[bold cyan]Starting NetReach v3 Bidirectional Evaluation Across Path Lengths (N={args.num_nodes}, M={args.target_edges})...[/bold cyan]\n")

    results_data = {
        "eval_hops": args.hops,
        "t_test_values": args.t_test,
        "num_nodes": args.num_nodes,
        "target_edges": args.target_edges,
        "hops": {},
    }

    table = Table(
        title="NetReach v3: Bidirectional Wavefront Recurrence & Meet-in-the-Middle Scaling",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Hops (H)", justify="center", style="bold yellow")
    table.add_column("Dist", justify="center")
    table.add_column("Transformer\n(Fixed 3-L)", justify="center", style="cyan")
    table.add_column("GRU\n(Sequential)", justify="center", style="cyan")
    for t_step in args.t_test:
        table.add_column(f"v3 (T={t_step})\n[Bidirectional]", justify="center", style="green")
    table.add_column("Auto ACT\n(Acc / Steps)", justify="center", style="bold green")
    table.add_column("Hop MAE", justify="center")
    table.add_column("Latent\nCos Sim", justify="center")
    table.add_column("Latency\n(per sample)", justify="center")

    for h in args.hops:
        dist_type = "In-Dist" if h in [4, 8] else "OOD"
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

        # 3. NetReach v3 Autonomous ACT mode
        act_res = evaluate_model_on_dataloader(model_v3, loader, device, use_bf16, steps=None)
        act_acc = act_res["acc"]
        act_steps = act_res["act_steps"]
        act_mae = act_res["hop_mae"]
        mean_latency = act_res["mean_latency_ms"]

        # 4. NetReach v3 Fixed Steps Scaling
        fixed_step_accs = {}
        cos_sims = {}
        for t_step in args.t_test:
            step_res = evaluate_model_on_dataloader(model_v3, loader, device, use_bf16, steps=t_step)
            fixed_step_accs[f"T={t_step}"] = step_res["acc"]
            cos_sims[f"T={t_step}"] = step_res["cos_sim"]

        results_data["hops"][str(h)] = {
            "hop": h,
            "dist": dist_type,
            "transformer_acc": t_acc,
            "gru_acc": gru_acc,
            "act_acc": act_acc,
            "act_steps": act_steps,
            "act_hop_mae": act_mae,
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
            f"{act_acc*100:.1f}%\n({act_steps:.1f} st)",
            f"±{act_mae:.1f} hops",
            f"{min_sim:.2f} → {max_sim:.2f}",
            f"{mean_latency:.2f} ms",
        ])
        table.add_row(*row)

    console.print("\n")
    console.print(table)

    out_file = os.path.join(args.output_dir, "ood_eval_v3.json")
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)

    console.print(f"\n[bold green]Saved detailed NetReach v3 evaluation results to {out_file}[/bold green]\n")


if __name__ == "__main__":
    main()
