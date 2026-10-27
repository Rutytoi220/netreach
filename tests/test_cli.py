"""
Unit tests for NetReach CLI interface.
"""

import sys
import json
import pytest
from netreach.cli import main


def test_cli_help(capsys, monkeypatch):
    """Verify netreach -h displays help message."""
    monkeypatch.setattr(sys, "argv", ["netreach", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "NetReach" in captured.out


def test_cli_audit_json(capsys, monkeypatch):
    """Verify netreach audit output with --json flag."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "netreach",
            "audit",
            "--graph",
            "examples/active_directory.json",
            "--src",
            "WORKSTATION-09.CORP",
            "--dst",
            "DC01.CORP",
            "--depth",
            "5",
            "10",
            "--device",
            "cpu",
            "--json",
        ],
    )
    main()
    captured = capsys.readouterr()
    output_data = json.loads(captured.out)

    assert output_data["source"] == "WORKSTATION-09.CORP"
    assert output_data["target"] == "DC01.CORP"
    assert "probabilities" in output_data
    assert "5" in output_data["probabilities"]
    assert "10" in output_data["probabilities"]
    assert output_data["shortest_hop_count"] == 9
    assert output_data["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW / UNREACHABLE")


def test_cli_benchmark_json(capsys, monkeypatch):
    """Verify netreach benchmark output with --json flag."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "netreach",
            "benchmark",
            "--samples",
            "4",
            "--device",
            "cpu",
            "--json",
        ],
    )
    main()
    captured = capsys.readouterr()
    bench = json.loads(captured.out)
    assert bench["tested_samples"] == 4
    assert bench["parameter_count"] == 501_793
    assert "mean_latency_ms" in bench


def test_cli_audit_missing_file(capsys, monkeypatch):
    """Verify netreach audit exits cleanly on missing graph file."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "netreach",
            "audit",
            "--graph",
            "non_existent_file.json",
            "--src",
            "A",
            "--dst",
            "B",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
