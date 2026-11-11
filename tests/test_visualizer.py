"""
Unit and Integration Tests for Interactive Cytoscape.js Visualizer (`netreach view`).
"""

import os
import re
import json
import subprocess
import sys
import pytest

from netreach.visualizer import generate_corridor_visualization
from netreach.remediation import HardenResult, Chokepoint


def test_generate_corridor_visualization_dom(tmp_path):
    """Verify HTML DOM structure, embedded Cytoscape payload, and HUD elements."""
    cp1 = Chokepoint(
        rank=1,
        source="WS-01",
        target="SRV-01",
        relation="AdminTo",
        ad_relation="CanRDP",
        baseline_prob=0.91,
        ablated_prob=0.01,
        delta_p=0.90,
        mitigation_cmd="Remove-LocalGroupMember ...",
        cypher_query="MATCH (u)-[r]->(v) DELETE r",
        description="Sever RDP access",
    )

    harden_res = HardenResult(
        source="WS-01",
        target="DC-01",
        baseline_risk_level="CRITICAL",
        baseline_probability=0.91,
        act_steps=6.0,
        bridge_node="JUMPBOX",
        attack_path=["WS-01", "SRV-01", "JUMPBOX", "DC-01"],
        path_relations=[
            {"source": "WS-01", "target": "SRV-01", "relation": "AdminTo", "ad_relation": "CanRDP"},
            {"source": "SRV-01", "target": "JUMPBOX", "relation": "HasSession", "ad_relation": "HasSession"},
            {"source": "JUMPBOX", "target": "DC-01", "relation": "AdminTo", "ad_relation": "GenericAll"},
        ],
        chokepoints=[cp1],
        analysis_latency_ms=85.2,
        subgraph_nodes=4,
        subgraph_edges=3,
    )

    out_file = str(tmp_path / "corridor.html")
    res_path = generate_corridor_visualization(harden_res, output_path=out_file)

    assert os.path.exists(res_path)
    with open(res_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    # Verify key DOM components
    assert "cytoscape.min.js" in html_text
    assert 'id="cy"' in html_text
    assert 'id="sidebar"' in html_text
    assert 'id="meter-fill"' in html_text
    assert 'id="chokepoints-list"' in html_text
    assert "WS-01" in html_text
    assert "DC-01" in html_text
    assert "JUMPBOX" in html_text

    # Extract and parse embedded JSON payload
    match = re.search(r"const PAYLOAD = (\{.*?\});\n", html_text, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))

    assert payload["telemetry"]["source"] == "WS-01"
    assert payload["telemetry"]["target"] == "DC-01"
    assert payload["telemetry"]["bridge_node"] == "JUMPBOX"
    assert len(payload["chokepoints"]) == 1
    assert payload["chokepoints"][0]["delta_p"] == 0.90
    assert len(payload["elements"]) >= 7  # 4 nodes + 3 edges


def test_cli_view_generation(tmp_path):
    """Test CLI 'netreach view' command."""
    graph_path = "examples/bloodhound_enterprise.json"
    out_html = str(tmp_path / "test_view.html")

    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "view",
        "--graph",
        graph_path,
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--output",
        out_html,
        "--depth",
        "6",
        "--top-k",
        "2",
        "--int8",
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    assert "Interactive corridor visualization generated" in res.stdout
    assert os.path.exists(out_html)

    with open(out_html, "r", encoding="utf-8") as f:
        content = f.read()
    assert "WORKSTATION-01.CORP.LOCAL" in content
    assert "DC01.CORP.LOCAL" in content
