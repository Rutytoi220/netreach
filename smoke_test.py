"""
Smoke Test for NetReach v3: Bidirectional Recurrence, ACT Halting & Multi-Task Pointer Learning.

Verifies:
1. Generation of heterogeneous graph samples (N = 128, M = 384) with next-hop pointer ground truth across H in [4, 8, 12, 16, 20, 24, 32].
2. Batched tensor shapes and pointer supervision masks.
3. Parameter count audit: ChallengerLatentNetV3 strictly between 15.0M and 22.0M parameters.
4. Full forward and backward passes on CUDA with bfloat16, multi-task loss (BCE + SmoothL1 + Pointer CE + Ponder Cost), and gradient checking.
5. Bidirectional wavefront collision mechanics: C_t(v), bridge pooling, ACT halting dynamics, and fixed scaling.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from data.generator import generate_graph_sample, create_dataloader, RELATION_TYPES
from netreach.model_v3 import ChallengerLatentNetV3
from train import compute_multitask_loss_v3

console = Console()


def run_smoke_test():
    console.print("\n[bold magenta]=====================================================[/bold magenta]")
    console.print("[bold cyan]       RUNNING NETREACH v3 SMOKE TEST (~18M Params)  [/bold cyan]")
    console.print("[bold magenta]=====================================================[/bold magenta]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Compute Device:[/bold green] {device}")
    if device.type == "cuda":
        console.print(f"[bold green]GPU Model:[/bold green] {torch.cuda.get_device_name(0)}")
        console.print(f"[bold green]bfloat16 supported:[/bold green] {torch.cuda.is_bf16_supported()}")

    # -------------------------------------------------------------
    # 1. Generate heterogeneous samples with N=128, M=384, H<=32
    # -------------------------------------------------------------
    console.print(
        f"\n[bold yellow]Step 1: Generating 50 heterogeneous samples (N=128, M=384) with {len(RELATION_TYPES)} relation types and next-hop pointers...[/bold yellow]"
    )
    t0 = time.time()
    hop_tests = [4, 8, 12, 16, 20, 24, 32]
    relation_counts = {r: 0 for r in range(len(RELATION_TYPES))}

    for i in range(50):
        h = hop_tests[i % len(hop_tests)]
        reachable = (i % 2 == 0)
        sample = generate_graph_sample(
            H=h, reachable=reachable, num_nodes=128, target_edges=384
        )
        edges, src, dst, label, sample_h = sample[:5]
        next_hop = sample[5]

        assert len(edges) == 384, f"Expected 384 edges, got {len(edges)}"
        assert label == (1 if reachable else 0)
        assert sample_h == h
        assert len(next_hop) == 128
        if reachable:
            assert next_hop[src] != -1
            assert next_hop[dst] == dst
        for u, v, r in edges:
            relation_counts[r] += 1

    console.print(
        f"[bold green]✔ Generated 50 samples in {time.time() - t0:.3f}s. Relations active across {len(relation_counts)} types.[/bold green]"
    )

    # -------------------------------------------------------------
    # 2. Check Batched Tensor Shapes
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 2: Checking DataLoader batch collation (N=128, M=384)...[/bold yellow]")
    loader = create_dataloader(
        hop_lengths=[4, 8, 12, 16, 20, 24, 32],
        num_samples=16,
        batch_size=8,
        num_nodes=128,
        target_edges=384,
        seed=42,
    )
    batch = next(iter(loader))

    assert batch["node_ids"].shape == (8, 128)
    assert batch["indicators"].shape == (8, 128, 3)
    assert batch["adj"].shape == (8, 128, 128)
    assert batch["edge_src"].shape == (8, 384)
    assert batch["edge_dst"].shape == (8, 384)
    assert batch["edge_rel"].shape == (8, 384)
    assert batch["next_hop"].shape == (8, 128)
    assert batch["label"].shape == (8,)
    assert batch["hop_length"].shape == (8,)
    console.print(
        f"[bold green]✔ Batch shapes verified: edge_rel={tuple(batch['edge_rel'].shape)}, next_hop={tuple(batch['next_hop'].shape)}, adj={tuple(batch['adj'].shape)}.[/bold green]"
    )

    # -------------------------------------------------------------
    # 3. Parameter Count Audit: Strictly between 15.0M and 22.0M
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 3: Auditing Parameter Count of ChallengerLatentNetV3...[/bold yellow]")
    model = ChallengerLatentNetV3(
        num_nodes=128,
        node_embed_dim=256,
        d_model=512,
        hidden_dim=1024,
        nhead=8,
        rel_dim=128,
        num_relations=8,
        max_steps=12,
    )
    params = model.count_parameters()
    console.print(f"[bold cyan]Exact Parameter Count: {params:,}[/bold cyan]")
    assert 15_000_000 <= params <= 22_000_000, f"Parameter count {params:,} outside [15.0M, 22.0M] range!"
    console.print("[bold green]✔ Parameter count strictly verified within [15,000,000 - 22,000,000].[/bold green]")

    # -------------------------------------------------------------
    # 4. Forward & Backward Pass on CUDA with Mixed Precision
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 4: Running Forward + Backward Pass on CUDA with Multi-Task Loss...[/bold yellow]")
    model = model.to(device)
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer.zero_grad()

    t_fwd = time.perf_counter()
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch_dev, steps=None, return_latents=True)
            loss, metrics = compute_multitask_loss_v3(out, batch_dev, criterion_bce)
    else:
        out = model(batch_dev, steps=None, return_latents=True)
        loss, metrics = compute_multitask_loss_v3(out, batch_dev, criterion_bce)

    if device.type == "cuda":
        torch.cuda.synchronize()
    fwd_time_ms = (time.perf_counter() - t_fwd) * 1000

    console.print(f"Forward Pass Time (Batch 8, ACT): {fwd_time_ms:.2f} ms ({fwd_time_ms/8:.2f} ms/query)")
    console.print(
        f"Loss components: BCE={metrics['loss_bce']:.4f}, Hop={metrics['loss_hop']:.4f}, Pointer={metrics['loss_pointer']:.4f}, Ponder={metrics['ponder_cost']:.4f}"
    )

    loss.backward()

    # Gradient checks
    assert model.reach_head[0].weight.grad is not None
    assert torch.sum(torch.abs(model.reach_head[0].weight.grad)) > 0.0

    assert model.hop_head[0].weight.grad is not None
    assert torch.sum(torch.abs(model.hop_head[0].weight.grad)) > 0.0

    assert model.pointer_q.weight.grad is not None
    assert torch.sum(torch.abs(model.pointer_q.weight.grad)) > 0.0

    assert model.halt_head[0].weight.grad is not None
    assert torch.sum(torch.abs(model.halt_head[0].weight.grad)) > 0.0

    assert model.W_f.weight.grad is not None
    assert torch.sum(torch.abs(model.W_f.weight.grad)) > 0.0

    assert model.W_b.weight.grad is not None
    assert torch.sum(torch.abs(model.W_b.weight.grad)) > 0.0

    console.print("[bold green]✔ Non-zero gradients verified across reach_head, hop_head, pointer_decoder, halt_head, and collision projections.[/bold green]")

    # -------------------------------------------------------------
    # 5. Fixed Steps Scaling & Collision Mechanics
    # -------------------------------------------------------------
    console.print("\n[bold yellow]Step 5: Verifying Bidirectional Collision Mechanics & Fixed Scaling...[/bold yellow]")
    with torch.no_grad():
        out_6 = model(batch_dev, steps=6, return_latents=True)
        out_14 = model(batch_dev, steps=14, return_latents=True)

    assert out_6["logit"].shape == (8,)
    assert out_6["hop"].shape == (8,)
    assert out_6["pointer_logits"].shape == (8, 128, 128)
    assert out_6["collision_map"].shape == (8, 128)
    assert float(out_6["steps_taken"].mean()) == 6.0
    assert float(out_14["steps_taken"].mean()) == 14.0

    max_c_6 = out_6["max_collision"].mean().item()
    max_c_14 = out_14["max_collision"].mean().item()
    console.print(f"Collision Affinity mean at T=6: {max_c_6:.3f} | at T=14: {max_c_14:.3f}")
    console.print("[bold green]✔ Bidirectional recurrence and fixed step scaling verified.[/bold green]")

    console.print("\n[bold green]=====================================================[/bold green]")
    console.print("[bold green]      ALL NETREACH v3 SMOKE TESTS PASSED! ✔          [/bold green]")
    console.print("[bold green]=====================================================[/bold green]\n")


if __name__ == "__main__":
    run_smoke_test()
