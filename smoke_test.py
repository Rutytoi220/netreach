"""
Smoke Test for Upgraded Algorithmic Generalization Benchmark.

Performs:
1. Generation of 100 graph samples with N=48 nodes across H in [3, 5, 8, 12, 16, 20].
2. Tensor shape verification across batched collations (N=48, M=96).
3. Verification of parameter counts for all 3 models (matched strictly between 400k and 600k).
4. Full forward and backward passes with 1 training epoch for all 3 models on CUDA with bfloat16.
5. Gated latent recurrence unrolling for T_test in [5, 10, 15, 20] and mathematical cosine similarity diagnostics.
"""

import time
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from data.generator import (
    generate_graph_sample,
    create_dataloader,
)
from models.baseline_transformer import BaselineTransformer
from models.baseline_gru import BaselineGRU
from models.challenger_latent_net import ChallengerLatentNet
from evaluate_ood import compute_pairwise_cosine_similarity

console = Console()


def run_smoke_test():
    console.print("\n[bold magenta]=====================================================[/bold magenta]")
    console.print("[bold cyan]       RUNNING UPGRADED SMOKE TEST (N=48)            [/bold cyan]")
    console.print("[bold magenta]=====================================================[/bold magenta]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Compute Device:[/bold green] {device}")
    if device.type == "cuda":
        console.print(f"[bold green]GPU Model:[/bold green] {torch.cuda.get_device_name(0)}")
        console.print(f"[bold green]bfloat16 supported:[/bold green] {torch.cuda.is_bf16_supported()}")

    # 1. Generate 100 samples across H in [3, 5, 8, 12, 16, 20]
    console.print("\n[bold yellow]Step 1: Generating 100 graph samples across H in [3, 5, 8, 12, 16, 20] (N=48)...[/bold yellow]")
    t0 = time.time()
    hop_tests = [3, 5, 8, 12, 16, 20]
    for i in range(100):
        h = hop_tests[i % len(hop_tests)]
        reachable = (i % 2 == 0)
        edges, src, dst, label, sample_h = generate_graph_sample(H=h, reachable=reachable, num_nodes=48, target_edges=96)
        assert len(edges) == 96, f"Expected 96 edges, got {len(edges)}"
        assert label == (1 if reachable else 0)
        assert sample_h == h
    console.print(f"[bold green]✔ Successfully generated 100 samples (N=48, M=96) in {time.time() - t0:.3f}s[/bold green]")

    # 2. Verify Tensor Shapes with DataLoader
    console.print("\n[bold yellow]Step 2: Verifying batched tensor shapes (N=48, M=96)...[/bold yellow]")
    loader = create_dataloader(hop_lengths=[3, 4, 5], num_samples=100, batch_size=32, shuffle=True, num_nodes=48, target_edges=96)
    batch = next(iter(loader))

    expected_shapes = {
        "node_ids": (32, 48),
        "indicators": (32, 48, 3),
        "adj": (32, 48, 48),
        "edge_src": (32, 96),
        "edge_dst": (32, 96),
        "src_idx": (32,),
        "dst_idx": (32,),
        "label": (32,),
        "hop_length": (32,),
    }

    table = Table(title="Batch Tensor Shapes Verification", header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Actual Shape", style="green")
    table.add_column("Expected Shape", style="yellow")
    table.add_column("Status", style="bold green")

    for k, exp in expected_shapes.items():
        act = tuple(batch[k].shape)
        assert act == exp, f"Shape mismatch for {k}: {act} vs {exp}"
        table.add_row(k, str(act), str(exp), "PASSED ✔")
    console.print(table)

    # 3. Model Parameter Count Verification [400K - 600K]
    console.print("\n[bold yellow]Step 3: Initializing models and verifying parameter budget [400K - 600K]...[/bold yellow]")
    models = {
        "BaselineTransformer": BaselineTransformer(num_nodes=50).to(device),
        "BaselineGRU": BaselineGRU(num_nodes=50).to(device),
        "ChallengerLatentNet": ChallengerLatentNet(num_nodes=50, train_steps=5).to(device),
    }

    param_table = Table(title="Parameter Budget Audit", header_style="bold magenta")
    param_table.add_column("Model Name", style="bold cyan")
    param_table.add_column("Parameter Count", justify="right", style="green")
    param_table.add_column("Allowed Range", justify="center", style="yellow")
    param_table.add_column("Compliance", justify="center", style="bold green")

    for name, m in models.items():
        count = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert 400_000 <= count <= 600_000, f"{name} count {count} is outside [400k, 600k]"
        param_table.add_row(name, f"{count:,}", "[400,000 - 600,000]", "STRICT COMPLIANCE ✔")
    console.print(param_table)

    # 4. Run 1 Training Epoch for All 3 Models
    console.print("\n[bold yellow]Step 4: Executing 1 Training Epoch on CUDA for all 3 models...[/bold yellow]")
    criterion = nn.BCEWithLogitsLoss()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    for name, model in models.items():
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        epoch_loss = 0.0
        correct = 0
        total = 0

        t_epoch = time.time()
        for b in loader:
            b_dev = {k: v.to(device) for k, v in b.items()}
            labels = b_dev["label"]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(b_dev)
                loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = epoch_loss / total
        acc = correct / total
        dt = time.time() - t_epoch
        console.print(f"[bold green]✔ {name}[/bold green] | Epoch 1 Loss: {avg_loss:.4f} | Accuracy: {acc*100:.1f}% | Time: {dt:.2f}s")

    # 5. Verify Test-time Recurrence Steps & Cosine Similarity Diagnostic
    console.print("\n[bold yellow]Step 5: Verifying Gated Latent Recurrence & Cosine Similarity Diagnostics at T in [5, 10, 15, 20]...[/bold yellow]")
    challenger = models["ChallengerLatentNet"]
    challenger.eval()
    b_dev = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        for t_test in [5, 10, 15, 20]:
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits_t, latents_t = challenger(b_dev, steps=t_test, return_latents=True)
            cos_sim = compute_pairwise_cosine_similarity(latents_t)
            is_oversmoothed = cos_sim >= 0.85
            status = "[red]OVER-SMOOTHED[/red]" if is_oversmoothed else "[green]HEALTHY DIVERSITY ✔[/green]"
            console.print(
                f"[bold green]✔ ChallengerLatentNet (T_test = {t_test:2d})[/bold green] | "
                f"Pairwise Cosine Sim: {cos_sim:.4f} | Status: {status}"
            )

    console.print("\n[bold magenta]=====================================================[/bold magenta]")
    console.print("[bold green]       UPGRADED SMOKE TEST COMPLETED SUCCESSFULLY!    [/bold green]")
    console.print("[bold magenta]=====================================================[/bold magenta]\n")


if __name__ == "__main__":
    run_smoke_test()
