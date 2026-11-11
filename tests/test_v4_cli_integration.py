"""
Integration tests for NetReach v0.5.0 CLI and Engine with ChallengerLatentNetV4.
"""

import sys
import json
import subprocess
import pytest
import torch

from netreach.engine import NetReachEngine, AuditResult
from netreach.model_v4 import ChallengerLatentNetV4


def test_engine_initialization_v4():
    engine = NetReachEngine(model_version="v4")
    assert isinstance(engine.model, ChallengerLatentNetV4)
    assert 35_000_000 <= engine.param_count <= 45_000_000


def test_engine_int8_quantization():
    engine = NetReachEngine(model_version="v4", use_int8=True)
    assert engine.use_int8 is True
    assert engine.device.type == "cpu"


def test_engine_audit_bloodhound():
    engine = NetReachEngine(model_version="v4")
    result = engine.audit(
        graph_source="examples/bloodhound_enterprise.json",
        source="WORKSTATION-01.CORP.LOCAL",
        target="DC01.CORP.LOCAL",
        depths=(4, 8),
    )
    assert isinstance(result, AuditResult)
    assert result.source == "WORKSTATION-01.CORP.LOCAL"
    assert result.target == "DC01.CORP.LOCAL"
    assert 4 in result.probabilities
    assert 8 in result.probabilities
    assert result.shortest_hop_count == 9
    assert len(result.ground_truth_path) == 10
    assert result.risk_level in ("HIGH", "CRITICAL")
    assert result.is_valid_walk is True
    assert len(result.path_relations) >= 8


def test_cli_audit_json_output():
    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "audit",
        "--graph",
        "examples/bloodhound_enterprise.json",
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--depth",
        "4",
        "8",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert out["target"] == "DC01.CORP.LOCAL"
    assert "probabilities" in out
    assert "shortest_hop_count" in out
    assert "path_relations" in out
    assert "act_steps" in out
    assert "peak_collision" in out
    assert out["shortest_hop_count"] == 9


def test_cli_audit_int8_cpu_flag():
    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "audit",
        "--graph",
        "examples/bloodhound_enterprise.json",
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--depth",
        "4",
        "--int8",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert "4" in out["probabilities"]
    assert out["is_valid_walk"] is True
