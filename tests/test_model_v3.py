"""
Unit tests for ChallengerLatentNetV3 model architecture, dual-wavefront meet-in-the-middle recurrence,
ACT halting, and next-hop pointer decoding.
"""

import pytest
import torch
import torch.nn.functional as F

from netreach.model_v3 import ChallengerLatentNetV3, RELATION_TYPES, RELATION_TO_ID
from data.generator import ReachabilityDataset, collate_graph_batch
from train import compute_multitask_loss_v3


def test_v3_parameter_budget():
    """Verify ChallengerLatentNetV3 strictly complies with 15.0M - 22.0M parameters."""
    model = ChallengerLatentNetV3(
        num_nodes=128,
        node_embed_dim=256,
        d_model=512,
        hidden_dim=1024,
        nhead=8,
        rel_dim=128,
        num_relations=8,
    )
    count = model.count_parameters()
    assert 15_000_000 <= count <= 22_000_000, f"Expected params in [15M, 22M], got {count}"
    assert count == 18_280_707


def test_v3_relation_types():
    """Verify the 6 defined directed relation types."""
    expected_relations = [
        "AdminTo",
        "MemberOf",
        "HasSession",
        "AllowedToDelegate",
        "OpenPort",
        "SubnetRoute",
    ]
    assert RELATION_TYPES == expected_relations
    assert len(RELATION_TO_ID) == 6


def test_v3_forward_act_cpu():
    """Verify forward ACT pass output shapes and bounds on CPU."""
    dataset = ReachabilityDataset(
        hop_lengths=[4, 8], num_samples=4, num_nodes=128, target_edges=384, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV3(num_nodes=128, max_steps=10)
    model.eval()

    with torch.no_grad():
        out = model(batch, steps=None, return_latents=True)

    assert "logit" in out
    assert "hop" in out
    assert "pointer_logits" in out
    assert "steps_taken" in out
    assert "ponder_cost" in out
    assert "collision_map" in out

    B = 2
    assert out["logit"].shape == (B,)
    assert out["hop"].shape == (B,)
    assert out["pointer_logits"].shape == (B, 128, 128)
    assert out["collision_map"].shape == (B, 128)
    assert out["steps_taken"].shape == (B,)

    assert (out["steps_taken"] >= 1.0).all()
    assert (out["steps_taken"] <= 10.0).all()


def test_v3_forward_fixed_steps():
    """Verify forward pass with fixed step unrolling."""
    dataset = ReachabilityDataset(
        hop_lengths=[4, 8], num_samples=4, num_nodes=128, target_edges=384, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV3(num_nodes=128, max_steps=14)
    model.eval()

    with torch.no_grad():
        out_6 = model(batch, steps=6, return_latents=True)
        out_12 = model(batch, steps=12, return_latents=True)

    assert out_6["logit"].shape == (2,)
    assert out_6["hop"].shape == (2,)
    assert out_6["pointer_logits"].shape == (2, 128, 128)
    assert float(out_6["steps_taken"].mean()) == 6.0

    assert out_12["logit"].shape == (2,)
    assert out_12["hop"].shape == (2,)
    assert out_12["pointer_logits"].shape == (2, 128, 128)
    assert float(out_12["steps_taken"].mean()) == 12.0


def test_v3_gradient_flow():
    """Verify backward gradient flow into all three heads and recurrent message passing layers."""
    dataset = ReachabilityDataset(
        hop_lengths=[4, 8], num_samples=4, num_nodes=128, target_edges=384, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV3(num_nodes=128, max_steps=6)
    model.train()

    out = model(batch, steps=None)
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    total_loss, metrics = compute_multitask_loss_v3(out, batch, criterion_bce)

    total_loss.backward()

    # Verify gradients exist and are non-zero across components
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

    assert model.layer1.W_q.weight.grad is not None
    assert torch.sum(torch.abs(model.layer1.W_q.weight.grad)) > 0.0
