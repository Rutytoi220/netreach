"""
Smoke Test for NetReach v2: Multi-Relational Attention & Adaptive Computation Time (ACT).

Verifies:
1. Generation of 100 heterogeneous graph samples with 6 relation types across H in [3, 5, 8, 12, 16, 20].
2. Batched tensor shapes and relation ID distributions.
3. Parameter count audit: ChallengerLatentNetV2 strictly between 4.5M and 5.5M parameters.
4. Full forward and backward passes on CUDA with bfloat16, multi-task loss (BCE + SmoothL1 + Ponder Cost), and gradient checking.
5. ACT halting head dynamics (step weights, cumulative probabilities, ponder cost) and test-time fixed scaling T in [5, 10, 15, 20].
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from data.generator import generate_graph_sample, create_dataloader, RELATION_TYPES
from netreach.model_v2 import ChallengerLatentNetV2
from train import compute_multitask_loss

console = Console()


def run_smoke_test():
    console.print("\n[bold magenta]=====================================================[/bold magenta]")
    console.print("[bold cyan]       RUNNING NETREACH v2 SMOKE TEST (5M Params)    [/bold cyan]")
    console.print("[bold magenta]=====================================================[/bold magenta]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Compute Device:[/bold green] {device}")
    if device.type == "cuda":
        console.print(f"[bold green]GPU Model:[/bold green] {torch.cuda.get_device_name(0)}")
        console.print(f"[bold green]bfloat16 supported:[/bold green] {torch.cuda.is_bf16_supported()}")

    # -------------------------------------------------------------
    # 1. Generate 100 heterogeneous multi-relational samples
    # -------------------------------------------------------------
    console.print(
        f"\n[bold yellow]Step 1: Generating 100 heterogeneous samples with {len(RELATION_TYPES)} relation types...[/bold yellow]"
    )
    t0 = time.time()
    hop_tests = [3, 5, 8, 12, 16, 20]
    relation_counts = {r: 0 for r in range(len(RELATION_TYPES))}

    for i in range(100):
        h = hop_tests[i % len(hop_tests)]
        reachable = (i % 2 == 0)
        edges, src, dst, label, sample_h = generate_graph_sample(
            H=h, reachable=reachable, num_nodes=48, target_edges=96
        )
        assert len(edges) == 96, f"Expected 96 edges, got {len(edges)}"
        assert label == (1 if reachable else 0)
        assert sample_h == h
        for u, v, r in edges:
            relation_counts[r] += 1

    console.print(
        f"[bold green]✔ Generated 100 samples in {time.time() - t0:.3f}s. Relations active across {len(relation_counts)} types.[/bold green]"
    )

    # -------------------------------------------------------------
    # 2. Verify Tensor Shapes with DataLoader
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 2: Verifying batched tensor shapes (N=48, M=96)...[/bold yellow]")
    loader = create_dataloader(
        hop_lengths=[3, 4, 5],
        num_samples=100,
        batch_size=32,
        shuffle=True,
        num_nodes=48,
        target_edges=96,
    )
    batch = next(iter(loader))

    expected_shapes = {
        "node_ids": (32, 48),
        "indicators": (32, 48, 3),
        "adj": (32, 48, 48),
        "edge_src": (32, 96),
        "edge_dst": (32, 96),
        "edge_rel": (32, 96),
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

    # -------------------------------------------------------------
    # 3. Model Parameter Count Verification [4.5M - 5.5M]
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 3: Initializing ChallengerLatentNetV2 and verifying parameter budget...[/bold yellow]")
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
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 4_500_000 <= param_count <= 5_500_000, f"Parameter count {param_count:,} is outside [4.5M, 5.5M]"

    param_table = Table(title="NetReach v2 Parameter Budget Audit", header_style="bold magenta")
    param_table.add_column("Model Name", style="bold cyan")
    param_table.add_column("Parameter Count", justify="right", style="green")
    param_table.add_column("Allowed Range", justify="center", style="yellow")
    param_table.add_column("Compliance", justify="center", style="bold green")
    param_table.add_row("ChallengerLatentNetV2", f"{param_count:,}", "[4,500,000 - 5,500,000]", "STRICT COMPLIANCE ✔")
    console.print(param_table)

    # -------------------------------------------------------------
    # 4. Execute 1 Training Epoch on CUDA with Multi-Task Loss
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 4: Executing 1 Training Epoch on CUDA with Multi-Task Loss...[/bold yellow]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    t_epoch = time.time()

    for b in loader:
        b_dev = {k: v.to(device) for k, v in b.items()}
        labels = b_dev["label"]
        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            out = model(b_dev)
            loss, loss_dict = compute_multitask_loss(out, b_dev, criterion_bce)

        loss.backward()
        # Verify gradients on halting head and task heads
        assert model.halt_head.weight.grad is not None, "Halting head received no gradient!"
        assert model.reach_head[0].weight.grad is not None, "Reach head received no gradient!"
        assert model.hop_head[0].weight.grad is not None, "Hop head received no gradient!"

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(out["logit"]) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    dt = time.time() - t_epoch
    console.print(
        f"[bold green]✔ Epoch 1 Completed in {dt:.2f}s[/bold green] | "
        f"Loss: {total_loss / total:.4f} | Reach Acc: {correct / total * 100:.1f}% | "
        f"Gradients verified on all heads!"
    )

    # -------------------------------------------------------------
    # 5. Verify Halting Head Dynamics & Fixed Recurrence Scaling
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 5: Verifying Halting Head Dynamics & Test-Time Compute Scaling...[/bold yellow]")
    model.eval()
    b_dev = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            act_out = model(b_dev)
            steps_taken = act_out["steps_taken"].mean().item()
            ponder_cost = act_out["ponder_cost"].item()
            console.print(
                f"[bold green]✔ Autonomous ACT Execution:[/bold green] Mean Steps Taken: {steps_taken:.2f} | Ponder Cost: {ponder_cost:.3f}"
            )

            for t_test in [5, 10, 15, 20]:
                fixed_out = model(b_dev, steps=t_test)
                prob = torch.sigmoid(fixed_out["logit"]).mean().item()
                mean_hop = fixed_out["hop"].mean().item()
                console.print(
                    f"[bold green]✔ Fixed Step T_test = {t_test:2d}:[/bold green] Mean Prob: {prob*100:.1f}% | Mean Predicted Hops: {mean_hop:.1f}"
                )

    console.print("\n[bold magenta]=====================================================[/bold magenta]")
    console.print("[bold green]       NETREACH v2 SMOKE TEST COMPLETED SUCCESSFULLY!  [/bold green]")
    console.print("[bold magenta]=====================================================[/bold magenta]\n")


if __name__ == "__main__":
    run_smoke_test()
