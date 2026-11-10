"""
Unit tests for ChallengerLatentNetV2 model architecture, ACT halting, and multi-relational attention.
"""

import pytest
import torch
import torch.nn.functional as F

from netreach.model_v2 import ChallengerLatentNetV2, RELATION_TYPES, RELATION_TO_ID
from data.generator import ReachabilityDataset, collate_graph_batch


def test_v2_parameter_budget():
    """Verify ChallengerLatentNetV2 strictly complies with 4.5M - 5.5M parameters."""
    model = ChallengerLatentNetV2(num_nodes=64)
    count = model.count_parameters()
    assert 4_500_000 <= count <= 5_500_000, f"Expected params in [4.5M, 5.5M], got {count}"
    assert count == 5_279_491

    # Also verify with num_nodes=48
    model_48 = ChallengerLatentNetV2(num_nodes=48)
    count_48 = model_48.count_parameters()
    assert 4_500_000 <= count_48 <= 5_500_000, f"Expected params in [4.5M, 5.5M], got {count_48}"
    assert count_48 == 5_277_443


def test_v2_relation_types():
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
    for rel in expected_relations:
        assert rel in RELATION_TO_ID


def test_v2_forward_act_cpu():
    """Verify forward ACT pass output shapes and probability bounds."""
    dataset = ReachabilityDataset(hop_lengths=[3, 5], num_samples=4, num_nodes=48, target_edges=96, seed=42)
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV2(num_nodes=48)
    model.eval()

    with torch.no_grad():
        out = model(batch, steps=None, return_latents=True)

    assert "logit" in out
    assert "hop" in out
    assert "steps_taken" in out
    assert "ponder_cost" in out
    assert "latents" in out

    B = 2
    assert out["logit"].shape == (B,)
    assert out["hop"].shape == (B,)
    assert out["steps_taken"].shape == (B,)
    assert out["ponder_cost"].ndim == 0  # scalar tensor
    assert out["latents"].shape == (B, 48, 384)

    assert (out["steps_taken"] >= 1.0).all()
    assert (out["steps_taken"] <= 20.0).all()
    assert float(out["ponder_cost"]) >= 0.0


def test_v2_forward_fixed_steps():
    """Verify forward pass under explicit fixed steps scaling."""
    dataset = ReachabilityDataset(hop_lengths=[3, 5], num_samples=4, num_nodes=48, target_edges=96, seed=42)
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV2(num_nodes=48)
    model.eval()

    with torch.no_grad():
        out_5 = model(batch, steps=5)
        out_15 = model(batch, steps=15)

    assert out_5["logit"].shape == (2,)
    assert out_5["hop"].shape == (2,)
    assert float(out_5["steps_taken"].mean()) == 5.0
    assert float(out_5["ponder_cost"]) == 0.0

    assert out_15["logit"].shape == (2,)
    assert out_15["hop"].shape == (2,)
    assert float(out_15["steps_taken"].mean()) == 15.0
    assert float(out_15["ponder_cost"]) == 0.0


def test_v2_gradient_flow():
    """Verify backward gradient flow into all three heads and recurrent parameters."""
    dataset = ReachabilityDataset(hop_lengths=[3, 5], num_samples=4, num_nodes=48, target_edges=96, seed=42)
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV2(num_nodes=48)
    model.train()

    out = model(batch, steps=None)

    bce_loss = F.binary_cross_entropy_with_logits(out["logit"], batch["label"])
    hop_loss = F.smooth_l1_loss(out["hop"], batch["hop_length"].float())
    ponder_cost = out["ponder_cost"]

    total_loss = 1.0 * bce_loss + 0.1 * hop_loss + 0.01 * ponder_cost
    total_loss.backward()

    # Verify gradients exist and are non-zero across components
    assert model.halt_head.weight.grad is not None
    assert torch.sum(torch.abs(model.halt_head.weight.grad)) > 0.0

    assert model.reach_head[0].weight.grad is not None
    assert torch.sum(torch.abs(model.reach_head[0].weight.grad)) > 0.0

    assert model.hop_head[0].weight.grad is not None
    assert torch.sum(torch.abs(model.hop_head[0].weight.grad)) > 0.0

    assert model.W_r.weight.grad is not None
    assert torch.sum(torch.abs(model.W_r.weight.grad)) > 0.0
