"""
Unit and Integration Tests for NetReach v0.6.0 Remediation & Chokepoint Analysis Engine.
"""

import json
import os
import subprocess
import sys
import pytest

from netreach.remediation import (
    generate_remediation_commands,
    find_critical_chokepoints,
    generate_markdown_report,
    Chokepoint,
    HardenResult,
)
from netreach.engine import NetReachEngine


def test_generate_remediation_commands_relations():
    """Verify mitigation generation for all supported Active Directory and network relations."""
    # MemberOf
    res_member = generate_remediation_commands("ALICE.ADMIN", "DOMAIN-ADMINS", "MemberOf", "MemberOf")
    assert "Remove-ADGroupMember" in res_member["powershell"]
    assert "DELETE r" in res_member["cypher"]
    assert "ALICE.ADMIN" in res_member["description"]

    # CanRDP (AdminTo variant)
    res_rdp = generate_remediation_commands("HELPDESK", "SRV-01", "AdminTo", "CanRDP")
    assert "Remote Desktop Users" in res_rdp["powershell"]
    assert "CanRDP" in res_rdp["cypher"]

    # ExecuteDCOM (AdminTo variant)
    res_dcom = generate_remediation_commands("BOB", "SRV-02", "AdminTo", "ExecuteDCOM")
    assert "Distributed COM Users" in res_dcom["powershell"]
    assert "ExecuteDCOM" in res_dcom["cypher"]

    # GenericAll / WriteDACL (AdminTo variant)
    res_acl = generate_remediation_commands("EVE", "DC01", "AdminTo", "GenericAll")
    assert "Get-Acl" in res_acl["powershell"]
    assert "GenericAll" in res_acl["cypher"]

    # DCSync / GetChanges (AdminTo variant)
    res_sync = generate_remediation_commands("MALLORY", "CORP.LOCAL", "AdminTo", "DCSync")
    assert "Revoke-ADReplicationPermissions" in res_sync["powershell"]

    # Default AdminTo
    res_admin = generate_remediation_commands("ADMIN_USER", "WORKSTATION-10", "AdminTo", "AdminTo")
    assert "Administrators" in res_admin["powershell"]

    # HasSession
    res_sess = generate_remediation_commands("JUMPBOX-01", "ALICE.ADMIN", "HasSession", "HasSession")
    assert "Protected Users" in res_sess["powershell"]
    assert "HasSession" in res_sess["cypher"]

    # AllowedToDelegate
    res_del = generate_remediation_commands("SVC-SQL", "TARGET-SRV", "AllowedToDelegate", "AllowedToDelegate")
    assert "TrustedForDelegation" in res_del["powershell"]
    assert "AllowedToDelegate" in res_del["cypher"]

    # OpenPort
    res_port = generate_remediation_commands("PIVOT-BOX", "INTERNAL-DB", "OpenPort", "OpenPort")
    assert "New-NetFirewallRule" in res_port["powershell"]

    # SubnetRoute
    res_route = generate_remediation_commands("DMZ-SUBNET", "PROD-VLAN", "SubnetRoute", "SubnetRoute")
    assert "SubnetRoute" in res_route["cypher"]


def test_find_critical_chokepoints_bloodhound(tmp_path):
    """Test fast neural edge ablation on real BloodHound enterprise JSON graph."""
    engine = NetReachEngine(device="cpu", use_int8=True)
    graph_path = "examples/bloodhound_enterprise.json"
    assert os.path.exists(graph_path)

    result = find_critical_chokepoints(
        graph_source=graph_path,
        src="WORKSTATION-01.CORP.LOCAL",
        dst="DC01.CORP.LOCAL",
        top_k=3,
        engine=engine,
        steps=6,
    )

    assert isinstance(result, HardenResult)
    assert result.source == "WORKSTATION-01.CORP.LOCAL"
    assert result.target == "DC01.CORP.LOCAL"
    assert result.baseline_probability > 0.70
    assert len(result.chokepoints) == 3

    # Ranking must be strictly descending by delta_p
    for i in range(len(result.chokepoints) - 1):
        assert result.chokepoints[i].delta_p >= result.chokepoints[i + 1].delta_p

    # Top chokepoint should significantly collapse attack path reachability
    top_cp = result.chokepoints[0]
    assert top_cp.delta_p > 0.50
    assert top_cp.ablated_prob < 0.25
    assert len(top_cp.mitigation_cmd) > 0
    assert len(top_cp.cypher_query) > 0


def test_engine_harden_method():
    """Verify that NetReachEngine.harden delegates cleanly to find_critical_chokepoints."""
    engine = NetReachEngine(device="cpu", use_int8=True)
    graph_path = "examples/bloodhound_enterprise.json"

    result = engine.harden(
        graph_source=graph_path,
        source="WORKSTATION-01.CORP.LOCAL",
        target="DC01.CORP.LOCAL",
        top_k=2,
        steps=6,
    )

    assert isinstance(result, HardenResult)
    assert len(result.chokepoints) == 2
    assert result.chokepoints[0].rank == 1
    assert result.chokepoints[1].rank == 2


def test_generate_markdown_report():
    """Test generating full markdown report from HardenResult."""
    cp1 = Chokepoint(
        rank=1,
        source="HELPDESK",
        target="SRV-FILES",
        relation="AdminTo",
        ad_relation="CanRDP",
        baseline_prob=0.92,
        ablated_prob=0.01,
        delta_p=0.91,
        mitigation_cmd="Remove-LocalGroupMember ...",
        cypher_query="MATCH ... DELETE r",
        description="Remove RDP permission",
    )
    result = HardenResult(
        source="WS-01",
        target="DC01",
        baseline_risk_level="CRITICAL",
        baseline_probability=0.92,
        act_steps=6.0,
        bridge_node="JUMPBOX",
        attack_path=["WS-01", "HELPDESK", "SRV-FILES", "DC01"],
        path_relations=[
            {"source": "WS-01", "target": "HELPDESK", "relation": "MemberOf", "ad_relation": "MemberOf"},
            {"source": "HELPDESK", "target": "SRV-FILES", "relation": "AdminTo", "ad_relation": "CanRDP"},
        ],
        chokepoints=[cp1],
        analysis_latency_ms=120.5,
        subgraph_nodes=4,
        subgraph_edges=3,
    )

    md = generate_markdown_report(result)
    assert "# 🛡️ NetReach Security Hardening & Chokepoint Remediation Report" in md
    assert "Executive Summary" in md
    assert "Critical Chokepoint Ranking" in md
    assert "Actionable Remediation Playbook" in md
    assert "Remove-LocalGroupMember" in md
    assert "MATCH ... DELETE r" in md


def test_cli_harden_json_and_export(tmp_path):
    """Test CLI netreach harden command with JSON output and report file export."""
    graph_path = "examples/bloodhound_enterprise.json"
    export_path = str(tmp_path / "hardening_report.md")

    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "harden",
        "--graph",
        graph_path,
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--depth",
        "6",
        "--top-k",
        "2",
        "--int8",
        "--export",
        export_path,
        "--json",
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    raw_json = res.stdout.strip()
    data = json.loads(raw_json)

    assert data["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert data["target"] == "DC01.CORP.LOCAL"
    assert data["baseline_probability"] > 0.70
    assert len(data["chokepoints"]) == 2
    assert data["chokepoints"][0]["delta_p"] > 0.50

    # Verify export file was written
    assert os.path.exists(export_path)
    with open(export_path, "r", encoding="utf-8") as f:
        exported_text = f.read()
    assert "# 🛡️ NetReach Security Hardening & Chokepoint Remediation Report" in exported_text
