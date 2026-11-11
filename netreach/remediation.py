"""
Automated Chokepoint Analysis & Remediation Engine for NetReach v0.6.0.

Provides fast neural edge ablation to discover minimal-impact defensive chokepoints
and generates actionable Active Directory PowerShell, host firewall, and BloodHound Cypher
mitigations to neutralize complex lateral movement attack paths.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import torch

from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch
from netreach.engine import NetReachEngine, AuditResult


@dataclass
class Chokepoint:
    """A critical defensive edge whose removal severs lateral movement reachability."""
    rank: int
    source: str
    target: str
    relation: str
    ad_relation: str
    baseline_prob: float
    ablated_prob: float
    delta_p: float
    mitigation_cmd: str
    cypher_query: str
    description: str


@dataclass
class HardenResult:
    """Structured result of a NetReach harden analysis."""
    source: str
    target: str
    baseline_risk_level: str
    baseline_probability: float
    act_steps: float
    bridge_node: Optional[str]
    attack_path: List[str]
    path_relations: List[Dict[str, str]]
    chokepoints: List[Chokepoint]
    analysis_latency_ms: float
    subgraph_nodes: int
    subgraph_edges: int


def generate_remediation_commands(
    source: str,
    target: str,
    canon_rel: str,
    ad_rel: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate tailored PowerShell, BloodHound Cypher, and operational guidance
    to sever a specified graph relationship.
    """
    ad_rel_clean = (ad_rel or canon_rel).strip()
    ad_rel_lower = ad_rel_clean.lower()
    canon_lower = canon_rel.lower()

    if canon_lower == "memberof":
        ps_cmd = f'Remove-ADGroupMember -Identity "{target}" -Members "{source}" -Confirm:$false'
        cypher = f'MATCH (u {{name: "{source}"}})-[r:MemberOf]->(g {{name: "{target}"}}) DELETE r'
        desc = f"Revoke active group membership of principal '{source}' from security group '{target}'."

    elif canon_lower == "adminto":
        if "rdp" in ad_rel_lower:
            ps_cmd = (
                f'Invoke-Command -ComputerName "{target}" -ScriptBlock {{ '
                f'Remove-LocalGroupMember -Group "Remote Desktop Users" -Member "{source}" }}'
            )
            cypher = f'MATCH (u {{name: "{source}"}})-[r:CanRDP]->(c {{name: "{target}"}}) DELETE r'
            desc = f"Remove '{source}' from the 'Remote Desktop Users' local group on host '{target}'."
        elif "dcom" in ad_rel_lower:
            ps_cmd = (
                f'# Revoke remote DCOM launch & activation permissions on host {target}\n'
                f'Invoke-Command -ComputerName "{target}" -ScriptBlock {{ '
                f'Remove-LocalGroupMember -Group "Distributed COM Users" -Member "{source}" }}'
            )
            cypher = f'MATCH (u {{name: "{source}"}})-[r:ExecuteDCOM]->(c {{name: "{target}"}}) DELETE r'
            desc = f"Revoke remote DCOM execution and activation rights for '{source}' on '{target}'."
        elif any(k in ad_rel_lower for k in ["genericall", "writedacl", "writeowner", "genericwrite", "owns"]):
            ps_cmd = (
                f'$targetAD = "{target}"\n'
                f'$acl = Get-Acl "AD:\\$targetAD"\n'
                f'$rule = $acl.Access | Where-Object {{ $_.IdentityReference -match "{source}" }}\n'
                f'$acl.RemoveAccessRule($rule)\n'
                f'Set-Acl "AD:\\$targetAD" $acl'
            )
            cypher = f'MATCH (u {{name: "{source}"}})-[r:{ad_rel_clean}]->(o {{name: "{target}"}}) DELETE r'
            desc = f"Purge Active Directory DACL entry granting '{ad_rel_clean}' privileges to '{source}' over '{target}'."
        elif any(k in ad_rel_lower for k in ["sync", "getchanges", "dcsync"]):
            ps_cmd = (
                f'# Revoke DS-Replication-Get-Changes extended access on domain head\n'
                f'Revoke-ADReplicationPermissions -Principal "{source}" -DomainObject "{target}"'
            )
            cypher = f'MATCH (u {{name: "{source}"}})-[r:{ad_rel_clean}]->(d {{name: "{target}"}}) DELETE r'
            desc = f"Revoke Directory Synchronization and Replication rights (DCSync) for '{source}' on domain '{target}'."
        else:
            ps_cmd = (
                f'Invoke-Command -ComputerName "{target}" -ScriptBlock {{ '
                f'Remove-LocalGroupMember -Group "Administrators" -Member "{source}" }}'
            )
            cypher = f'MATCH (u {{name: "{source}"}})-[r:AdminTo]->(c {{name: "{target}"}}) DELETE r'
            desc = f"Strip local Administrator privileges for account '{source}' on endpoint '{target}'."

    elif canon_lower == "hassession":
        ps_cmd = (
            f'# Terminate active interactive session and enroll account in Protected Users\n'
            f'Invoke-Command -ComputerName "{source}" -ScriptBlock {{ '
            f'query session | ForEach-Object {{ logoff ($_ -split "\\s+")[2] }} }};\n'
            f'Add-ADGroupMember -Identity "Protected Users" -Members "{target}"'
        )
        cypher = f'MATCH (c {{name: "{source}"}})-[r:HasSession]->(u {{name: "{target}"}}) DELETE r'
        desc = f"Terminate cached logon credentials on '{source}' and enforce Protected Users policy for '{target}'."

    elif canon_lower == "allowedtodelegate":
        ps_cmd = (
            f'Set-ADAccountControl -Identity "{source}" -TrustedForDelegation $false;\n'
            f'Set-ADUser -Identity "{source}" -Clear msDS-AllowedToDelegateTo'
        )
        cypher = f'MATCH (u {{name: "{source}"}})-[r:AllowedToDelegate]->(t {{name: "{target}"}}) DELETE r'
        desc = f"Disable unconstrained/constrained Kerberos delegation for principal '{source}' targeting '{target}'."

    elif canon_lower == "openport":
        ps_cmd = (
            f'Invoke-Command -ComputerName "{target}" -ScriptBlock {{ '
            f'New-NetFirewallRule -DisplayName "BlockPivot_{source}" -Direction Inbound -Action Block '
            f'-RemoteAddress (Resolve-DnsName "{source}").IPAddress }}'
        )
        cypher = f'MATCH (u {{name: "{source}"}})-[r:OpenPort]->(t {{name: "{target}"}}) DELETE r'
        desc = f"Apply host-based firewall policy on '{target}' blocking inbound connections from pivot host '{source}'."

    elif canon_lower == "subnetroute":
        ps_cmd = f'# Apply network microsegmentation / router ACL isolating {source} from {target}'
        cypher = f'MATCH (u {{name: "{source}"}})-[r:SubnetRoute]->(t {{name: "{target}"}}) DELETE r'
        desc = f"Update network gateway / VLAN ACL to drop route pivoting between '{source}' and '{target}'."

    else:
        ps_cmd = f'# Revoke relationship {ad_rel_clean} from {source} to {target}'
        cypher = f'MATCH (u {{name: "{source}"}})-[r]->(v {{name: "{target}"}}) DELETE r'
        desc = f"Sever relationship '{ad_rel_clean}' between '{source}' and '{target}'."

    return {
        "powershell": ps_cmd,
        "cypher": cypher,
        "description": desc,
    }


def find_critical_chokepoints(
    graph_source: Union[str, nx.DiGraph, List[Dict[str, Any]], Dict[str, Any]],
    src: str,
    dst: str,
    top_k: int = 3,
    engine: Optional[NetReachEngine] = None,
    steps: Optional[int] = None,
) -> HardenResult:
    """
    Perform Fast Neural Edge Ablation along the attack corridor to identify
    critical defensive chokepoints that collapse reachability with minimal operational friction.

    Args:
        graph_source: Path to graph file or existing NetworkX DiGraph.
        src: Attacker entry point asset (e.g. compromised workstation).
        dst: High-value objective asset (e.g. Domain Controller).
        top_k: Number of highest-impact chokepoints to return.
        engine: Existing NetReachEngine instance (or None to instantiate).
        steps: Recurrent steps T for ablation evaluations (defaults to baseline ACT steps or 8).

    Returns:
        HardenResult containing baseline metrics, ranked chokepoint edges, and mitigations.
    """
    t0 = time.perf_counter()

    if engine is None:
        engine = NetReachEngine()

    # 1. Run baseline audit to establish baseline reachability and attack corridor
    eval_depths = (steps,) if steps else (4, 8)
    audit_res = engine.audit(
        graph_source=graph_source,
        source=src,
        target=dst,
        depths=eval_depths,
    )

    baseline_step = steps if steps else (8 if 8 in audit_res.probabilities else max(audit_res.probabilities.keys()))
    baseline_prob = audit_res.probabilities[baseline_step]
    act_steps = audit_res.act_steps

    G = load_graph(graph_source)
    # Normalize node names
    src_clean = audit_res.source
    dst_clean = audit_res.target

    subG, meta = extract_subgraph(G, source=src_clean, target=dst_clean, max_nodes=256, max_edges=768)

    # 2. Extract candidate edges to ablate
    candidate_edges: List[Tuple[str, str, str, str]] = []
    seen_pairs = set()

    # Priority 1: Edges directly on the reconstructed/ground-truth attack path
    if audit_res.path_relations:
        for p_edge in audit_res.path_relations:
            u, v = p_edge["source"], p_edge["target"]
            c_rel = p_edge.get("relation", "AdminTo")
            ad_rel = p_edge.get("ad_relation", c_rel)
            candidate_edges.append((u, v, c_rel, ad_rel))
            seen_pairs.add((u, v))

    # Priority 2: Any edges in the corridor subgraph
    for u, v, data in subG.edges(data=True):
        if (u, v) not in seen_pairs:
            c_rel = data.get("relation", "AdminTo")
            ad_rel = data.get("ad_relation", c_rel)
            candidate_edges.append((u, v, c_rel, ad_rel))
            seen_pairs.add((u, v))

    # 3. Fast Neural Edge Ablation
    ablation_scores: List[Tuple[float, float, str, str, str, str]] = []

    with torch.no_grad():
        for u, v, c_rel, ad_rel in candidate_edges:
            # Create ablated corridor graph without edge (u, v)
            abl_subG = subG.copy()
            if abl_subG.has_edge(u, v):
                abl_subG.remove_edge(u, v)

            abl_batch = prepare_graph_batch(
                abl_subG,
                source=src_clean,
                target=dst_clean,
                target_nodes=256,
                target_edges=768,
                device=engine.device,
            )

            # Evaluate reachability drop under identical compute depth
            out = engine.model(abl_batch, steps=baseline_step)
            ablated_prob = torch.sigmoid(out["logit"][0]).item()
            delta_p = baseline_prob - ablated_prob

            ablation_scores.append((delta_p, ablated_prob, u, v, c_rel, ad_rel))

    # 4. Rank edges by reachability collapse ΔP descending
    ablation_scores.sort(key=lambda x: x[0], reverse=True)

    chokepoints: List[Chokepoint] = []
    for rank, (delta_p, abl_p, u, v, c_rel, ad_rel) in enumerate(ablation_scores[:top_k], 1):
        remediation = generate_remediation_commands(u, v, c_rel, ad_rel)
        chokepoints.append(
            Chokepoint(
                rank=rank,
                source=u,
                target=v,
                relation=c_rel,
                ad_relation=ad_rel,
                baseline_prob=baseline_prob,
                ablated_prob=abl_p,
                delta_p=delta_p,
                mitigation_cmd=remediation["powershell"],
                cypher_query=remediation["cypher"],
                description=remediation["description"],
            )
        )

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return HardenResult(
        source=src_clean,
        target=dst_clean,
        baseline_risk_level=audit_res.risk_level,
        baseline_probability=baseline_prob,
        act_steps=act_steps,
        bridge_node=audit_res.bridge_node,
        attack_path=audit_res.neural_path or audit_res.ground_truth_path,
        path_relations=audit_res.path_relations,
        chokepoints=chokepoints,
        analysis_latency_ms=latency_ms,
        subgraph_nodes=meta["extracted_nodes"],
        subgraph_edges=meta["extracted_edges"],
    )


def generate_markdown_report(result: HardenResult, graph_name: str = "Enterprise Graph") -> str:
    """
    Generate an executive incident remediation and attack path mitigation report in Markdown.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# 🛡️ NetReach Security Hardening & Chokepoint Remediation Report")
    md.append(f"\n**Target Assessment:** Lateral Movement Corridor between `{result.source}` and `{result.target}`  ")
    md.append(f"**Generated:** {now_str} | **Engine:** NetReach v0.6.0 (ChallengerLatentNetV4 38M)  ")
    md.append(f"**Baseline Assessment:** **{result.baseline_risk_level}** ({result.baseline_probability*100:.1f}% Reachability)\n")

    md.append(f"## 1. Executive Summary")
    md.append(
        f"A deep attack path was identified originating from compromised entry asset "
        f"`{result.source}` to high-value enterprise asset `{result.target}`. "
        f"Neural wavefront collision analysis identified **{len(result.chokepoints)} strategic defensive chokepoints**. "
        f"Severing the top-ranked chokepoint reduces lateral movement viability by **{result.chokepoints[0].delta_p*100:+.1f}%** "
        f"(collapsing reachability from {result.baseline_probability*100:.1f}% down to {result.chokepoints[0].ablated_prob*100:.1f}%)."
        if result.chokepoints
        else f"No viable multi-hop attack path detected between `{result.source}` and `{result.target}`."
    )

    md.append(f"\n## 2. Attack Corridor & Telemetry")
    md.append(f"- **Source Asset:** `{result.source}`")
    md.append(f"- **Target Asset:** `{result.target}`")
    md.append(f"- **Corridor Dimensions:** {result.subgraph_nodes} nodes, {result.subgraph_edges} edges")
    md.append(f"- **Collision Bridge Node ($v^*$):** `{result.bridge_node or 'None'}`")
    md.append(f"- **Halting Recurrence ($T$):** {result.act_steps:.1f} steps")
    md.append(f"- **Analysis Latency:** {result.analysis_latency_ms:.2f} ms")

    if result.path_relations:
        md.append(f"\n### Discovered Attack Path ({len(result.path_relations)} Hops):")
        md.append("```text")
        for i, edge in enumerate(result.path_relations, 1):
            rel_label = edge.get("ad_relation", edge.get("relation", "AdminTo"))
            md.append(f"{i:2d}. [{edge['source']}] ──({rel_label})──► [{edge['target']}]")
        md.append("```")

    md.append(f"\n## 3. Critical Chokepoint Ranking")
    md.append(f"Edges ranked by reachability collapse ($\Delta P = P_{{baseline}} - P_{{ablated}}$) via neural ablation:\n")
    md.append("| Rank | Source Principal | Target Asset | Relation | Ablated Reachability | Impact ($\Delta P$) |")
    md.append("| :---: | :--- | :--- | :---: | :---: | :---: |")

    for cp in result.chokepoints:
        md.append(
            f"| **#{cp.rank}** | `{cp.source}` | `{cp.target}` | `{cp.ad_relation}` | "
            f"{cp.ablated_prob*100:.1f}% | **{cp.delta_p*100:+.1f}%** |"
        )

    md.append(f"\n## 4. Actionable Remediation Playbook")
    for cp in result.chokepoints:
        md.append(f"### Priority #{cp.rank}: Sever `{cp.source}` ──({cp.ad_relation})──► `{cp.target}`")
        md.append(f"> **Remediation Objective:** {cp.description}  ")
        md.append(f"> **Reachability Impact:** Reduces attack path probability by **{cp.delta_p*100:.1f}%** (to {cp.ablated_prob*100:.1f}%).\n")

        md.append(f"**PowerShell / Administrative Remediation:**")
        md.append(f"```powershell\n{cp.mitigation_cmd}\n```\n")

        md.append(f"**BloodHound / Neo4j Simulation Query:**")
        md.append(f"```cypher\n{cp.cypher_query}\n```\n")

    md.append(f"## 5. Verification & Rollback Procedures")
    md.append(
        "1. **Simulation:** Execute the Cypher queries in BloodHound CE to visually confirm path severing.\n"
        "2. **Deployment:** Apply the PowerShell mitigations in a staged staging/canary administrative unit.\n"
        "3. **Validation:** Re-run `netreach harden --graph <updated_export> --src ... --dst ...` to verify that reachability falls below 10%.\n"
    )

    return "\n".join(md)
