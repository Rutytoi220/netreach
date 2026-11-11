"""
Interactive Standalone Visualizer for NetReach v0.7.0 (`netreach view`).

Generates self-contained, interactive HTML/JavaScript graph visualizations
using Cytoscape.js to render lateral movement attack corridors, collision bridges,
relation edge badges, and real-time interactive chokepoint severance simulations.
"""

import os
import json
import networkx as nx
from typing import Dict, List, Optional, Any, Union

from netreach.remediation import HardenResult, Chokepoint
from netreach.engine import NetReachEngine


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__</title>
  <!-- Cytoscape.js for interactive graph visualization -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #131b2e;
      --bg-panel: rgba(19, 27, 46, 0.85);
      --border-color: #1e293b;
      --accent-cyan: #00d2ff;
      --accent-red: #ef4444;
      --accent-yellow: #f59e0b;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    body {
      background-color: var(--bg-primary);
      color: var(--text-main);
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      display: flex;
    }

    #cy {
      flex: 1;
      height: 100vh;
      background: radial-gradient(circle at 50% 50%, #111827 0%, #0a0e17 100%);
    }

    /* Sidebar HUD */
    #sidebar {
      width: 440px;
      height: 100vh;
      background: var(--bg-panel);
      backdrop-filter: blur(12px);
      border-left: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      z-index: 10;
      box-shadow: -4px 0 24px rgba(0, 0, 0, 0.5);
    }

    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid var(--border-color);
      background: rgba(10, 14, 23, 0.7);
    }

    .badge-risk {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .risk-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .risk-high { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

    .sidebar-content {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    }

    .section-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-cyan);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .telemetry-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 20px;
    }

    .telemetry-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .telemetry-row:last-child { margin-bottom: 0; }
    .telemetry-label { color: var(--text-muted); }
    .telemetry-val { font-weight: 600; font-family: monospace; }

    /* Interactive Reachability Meter */
    .meter-container {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .meter-bar-bg {
      height: 10px;
      background: #1e293b;
      border-radius: 5px;
      overflow: hidden;
      position: relative;
      margin-top: 6px;
    }
    .meter-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
      transition: width 0.4s ease, background 0.4s ease;
    }

    /* Chokepoint Card */
    .chokepoint-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
      transition: border-color 0.2s, transform 0.2s;
      cursor: pointer;
    }
    .chokepoint-card:hover {
      border-color: var(--accent-cyan);
      transform: translateY(-1px);
    }
    .chokepoint-card.severed {
      border-color: var(--accent-red);
      background: rgba(239, 68, 68, 0.08);
    }

    .cp-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .cp-rank {
      font-weight: 700;
      color: var(--accent-yellow);
      font-size: 13px;
    }
    .cp-impact {
      font-weight: 700;
      color: var(--accent-green);
      font-size: 13px;
    }
    .cp-edge {
      font-size: 12px;
      font-family: monospace;
      color: var(--text-main);
      word-break: break-all;
      margin-bottom: 8px;
    }
    .sever-btn {
      width: 100%;
      padding: 6px 12px;
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid #ef4444;
      color: #f87171;
      border-radius: 4px;
      font-weight: 600;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    .sever-btn:hover {
      background: #ef4444;
      color: #fff;
    }
    .sever-btn.restored {
      background: rgba(16, 185, 129, 0.2);
      border-color: #10b981;
      color: #34d399;
    }
    .sever-btn.restored:hover {
      background: #10b981;
      color: #fff;
    }

    /* Path Hop Step */
    .hop-step {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 12px;
      font-family: monospace;
    }
    .hop-num {
      color: var(--text-muted);
      width: 20px;
      font-weight: 700;
    }
    .hop-edge-desc {
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Floating Toolbar */
    .floating-tools {
      position: absolute;
      top: 20px;
      left: 20px;
      display: flex;
      gap: 8px;
      z-index: 10;
    }
    .tool-btn {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      backdrop-filter: blur(8px);
      transition: background 0.2s, border-color 0.2s;
    }
    .tool-btn:hover {
      background: #1e293b;
      border-color: var(--accent-cyan);
    }

    /* Detail Modal / Code Inspector */
    .code-box {
      background: #050811;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 10px;
      font-family: monospace;
      font-size: 11px;
      color: #38bdf8;
      overflow-x: auto;
      margin-top: 8px;
      white-space: pre-wrap;
      word-break: break-all;
    }

    /* Legend */
    .legend {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--bg-panel);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      gap: 16px;
      font-size: 12px;
      z-index: 10;
    }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
  </style>
</head>
<body>

  <!-- Floating Controls -->
  <div class="floating-tools">
    <button class="tool-btn" onclick="cy.fit(null, 50)">⛶ Fit View</button>
    <button class="tool-btn" onclick="cy.layout({name: 'concentric', animate: true}).run()">◎ Concentric</button>
    <button class="tool-btn" onclick="cy.layout({name: 'breadthfirst', directed: true, animate: true}).run()">⚡ Hierarchical</button>
    <button class="tool-btn" onclick="resetAllSeverances()">↺ Reset All Edges</button>
  </div>

  <!-- Legend -->
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div> Source Asset</div>
    <div class="legend-item"><div class="legend-dot" style="background:#00d2ff; box-shadow: 0 0 8px #00d2ff;"></div> Bridge Node (v*)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div> Crown Jewel</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6;"></div> Attack Path</div>
    <div class="legend-item"><div class="legend-dot" style="background:#475569;"></div> Corridor Node</div>
  </div>

  <!-- Cytoscape Graph Canvas -->
  <div id="cy"></div>

  <!-- Interactive Sidebar HUD -->
  <div id="sidebar">
    <div class="sidebar-header">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h2 style="font-size: 16px; font-weight: 700; color: #fff;">⚡ NetReach Corridor</h2>
        <span id="risk-badge" class="badge-risk risk-critical">CRITICAL</span>
      </div>
      <p style="font-size: 12px; color: var(--text-muted); word-break: break-all;" id="assessment-target">
        Lateral Movement Assessment
      </p>
    </div>

    <div class="sidebar-content">
      <!-- Live Reachability Telemetry -->
      <div class="section-title">Live Telemetry & Simulation</div>
      <div class="telemetry-card">
        <div class="telemetry-row">
          <span class="telemetry-label">Source Asset</span>
          <span class="telemetry-val" id="val-src" style="color: #f87171;">-</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Crown Jewel</span>
          <span class="telemetry-val" id="val-dst" style="color: #fbbf24;">-</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Collision Bridge (v*)</span>
          <span class="telemetry-val" id="val-bridge" style="color: #00d2ff;">-</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">ACT Recurrence (T)</span>
          <span class="telemetry-val" id="val-act">-</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Analysis Latency</span>
          <span class="telemetry-val" id="val-latency">-</span>
        </div>

        <div class="meter-container">
          <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span class="telemetry-label">Attack Path Viability</span>
            <span class="telemetry-val" id="val-prob" style="font-size: 14px; font-weight: 700; color: #ef4444;">91.5%</span>
          </div>
          <div class="meter-bar-bg">
            <div class="meter-bar-fill" id="meter-fill" style="width: 91.5%;"></div>
          </div>
        </div>
      </div>

      <!-- Critical Chokepoints (Interactive Severance) -->
      <div class="section-title">Critical Chokepoints (Ranked by ΔP)</div>
      <div id="chokepoints-list"></div>

      <!-- Attack Path Steps -->
      <div class="section-title" style="margin-top: 20px;">Reconstructed Attack Corridor</div>
      <div id="path-hops-list"></div>

      <!-- Selected Mitigation Playbook -->
      <div class="section-title" style="margin-top: 20px;">Selected Mitigation Playbook</div>
      <div id="mitigation-details" class="telemetry-card">
        <p style="font-size: 12px; color: var(--text-muted);">Click an edge or chokepoint above to inspect PowerShell, Cloud CLI, and Cypher mitigations.</p>
      </div>
    </div>
  </div>

  <script>
    // Embedded Graph Payload from NetReach v0.7.0
    const PAYLOAD = __PAYLOAD_JSON__;

    const baselineProb = PAYLOAD.telemetry.baseline_probability;
    let currentProb = baselineProb;
    const severedEdges = new Set();

    // Populate HUD
    document.getElementById("val-src").textContent = PAYLOAD.telemetry.source;
    document.getElementById("val-dst").textContent = PAYLOAD.telemetry.target;
    document.getElementById("val-bridge").textContent = PAYLOAD.telemetry.bridge_node || "None";
    document.getElementById("val-act").textContent = PAYLOAD.telemetry.act_steps.toFixed(1) + " steps";
    document.getElementById("val-latency").textContent = PAYLOAD.telemetry.analysis_latency_ms.toFixed(1) + " ms";
    document.getElementById("assessment-target").textContent = PAYLOAD.telemetry.source + " ➔ " + PAYLOAD.telemetry.target;
    updateReachabilityMeter(baselineProb);

    // Initialize Cytoscape
    const cy = cytoscape({
      container: document.getElementById('cy'),
      elements: PAYLOAD.elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'color': '#f8fafc',
            'font-size': '11px',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'background-color': 'data(color)',
            'width': 'data(size)',
            'height': 'data(size)',
            'border-width': 2,
            'border-color': '#334155',
            'text-outline-color': '#0a0e17',
            'text-outline-width': 2
          }
        },
        {
          selector: 'node[role = "bridge"]',
          style: {
            'border-color': '#00d2ff',
            'border-width': 4,
            'shadow-blur': 15,
            'shadow-color': '#00d2ff',
            'shadow-opacity': 0.8
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(width)',
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(relation)',
            'font-size': '10px',
            'color': '#94a3b8',
            'text-rotation': 'autorotate',
            'text-margin-y': -8,
            'text-outline-color': '#0a0e17',
            'text-outline-width': 2
          }
        },
        {
          selector: 'edge[is_chokepoint = 1]',
          style: {
            'line-style': 'dashed',
            'line-dash-pattern': [6, 3],
            'line-color': '#f59e0b',
            'target-arrow-color': '#f59e0b',
            'width': 3
          }
        },
        {
          selector: 'edge.severed',
          style: {
            'line-color': '#ef4444',
            'target-arrow-color': '#ef4444',
            'opacity': 0.35,
            'line-style': 'dotted',
            'label': 'SEVERED ✕'
          }
        }
      ],
      layout: {
        name: 'breadthfirst',
        directed: true,
        roots: [`[id = "${PAYLOAD.telemetry.source}"]`],
        padding: 60,
        spacingFactor: 1.5
      }
    });

    // Populate Chokepoints
    const cpContainer = document.getElementById("chokepoints-list");
    PAYLOAD.chokepoints.forEach(cp => {
      const card = document.createElement("div");
      card.className = "chokepoint-card";
      card.id = `cp-card-${cp.edge_id}`;
      card.innerHTML = `
        <div class="cp-header">
          <span class="cp-rank">#${cp.rank} Critical Chokepoint</span>
          <span class="cp-impact">ΔP: +${(cp.delta_p * 100).toFixed(1)}%</span>
        </div>
        <div class="cp-edge">${cp.source} ──(${cp.ad_relation})──► ${cp.target}</div>
        <button class="sever-btn" id="btn-${cp.edge_id}" onclick="toggleSeverEdge('${cp.edge_id}', ${cp.ablated_prob}, event)">
          ✂ Sever Edge (Simulation)
        </button>
      `;
      card.addEventListener("click", () => showMitigationPlaybook(cp));
      cpContainer.appendChild(card);
    });

    // Populate Path Steps
    const hopsContainer = document.getElementById("path-hops-list");
    PAYLOAD.path_hops.forEach((hop, idx) => {
      const step = document.createElement("div");
      step.className = "hop-step";
      step.innerHTML = `
        <span class="hop-num">${idx + 1}.</span>
        <span class="hop-edge-desc" title="${hop.source} ──(${hop.relation})──► ${hop.target}">
          ${hop.source} ──(${hop.relation})──► ${hop.target}
        </span>
      `;
      hopsContainer.appendChild(step);
    });

    // Edge click handler in canvas
    cy.on('tap', 'edge', function(evt){
      const edge = evt.target;
      const edgeId = edge.id();
      const cp = PAYLOAD.chokepoints.find(c => c.edge_id === edgeId);
      if (cp) {
        showMitigationPlaybook(cp);
      }
    });

    // Toggle edge severance simulation
    window.toggleSeverEdge = function(edgeId, ablatedProb, evt) {
      if (evt) evt.stopPropagation();
      const edge = cy.getElementById(edgeId);
      const btn = document.getElementById(`btn-${edgeId}`);
      const card = document.getElementById(`cp-card-${edgeId}`);

      if (severedEdges.has(edgeId)) {
        // Restore edge
        severedEdges.delete(edgeId);
        edge.removeClass('severed');
        if (btn) {
          btn.textContent = "✂ Sever Edge (Simulation)";
          btn.className = "sever-btn";
        }
        if (card) card.classList.remove("severed");
        recalculateLiveProbability();
      } else {
        // Sever edge
        severedEdges.add(edgeId);
        edge.addClass('severed');
        if (btn) {
          btn.textContent = "↺ Restore Edge";
          btn.className = "sever-btn restored";
        }
        if (card) card.classList.add("severed");
        recalculateLiveProbability();
      }
    };

    function recalculateLiveProbability() {
      if (severedEdges.size === 0) {
        updateReachabilityMeter(baselineProb);
      } else {
        // If at least one top chokepoint severed, collapse probability to minimum ablated
        let minProb = baselineProb;
        severedEdges.forEach(eid => {
          const cp = PAYLOAD.chokepoints.find(c => c.edge_id === eid);
          if (cp && cp.ablated_prob < minProb) {
            minProb = cp.ablated_prob;
          }
        });
        updateReachabilityMeter(minProb);
      }
    }

    function updateReachabilityMeter(prob) {
      currentProb = prob;
      const pct = (prob * 100).toFixed(1);
      const valElem = document.getElementById("val-prob");
      const fillElem = document.getElementById("meter-fill");
      const badgeElem = document.getElementById("risk-badge");

      valElem.textContent = pct + "%";
      fillElem.style.width = pct + "%";

      if (prob >= 0.70) {
        valElem.style.color = "#ef4444";
        fillElem.style.background = "#ef4444";
        badgeElem.textContent = "CRITICAL";
        badgeElem.className = "badge-risk risk-critical";
      } else if (prob >= 0.30) {
        valElem.style.color = "#f59e0b";
        fillElem.style.background = "#f59e0b";
        badgeElem.textContent = "MODERATE";
        badgeElem.className = "badge-risk risk-high";
      } else {
        valElem.style.color = "#10b981";
        fillElem.style.background = "#10b981";
        badgeElem.textContent = "NEUTRALIZED";
        badgeElem.className = "badge-risk risk-low";
      }
    }

    function showMitigationPlaybook(cp) {
      const box = document.getElementById("mitigation-details");
      box.innerHTML = `
        <div style="font-weight: 700; font-size: 13px; color: #fff; margin-bottom: 6px;">
          ${cp.source} ➔ ${cp.target}
        </div>
        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">${cp.description}</p>
        <span style="font-size: 11px; font-weight: 600; color: #f59e0b;">PowerShell / CLI Mitigation:</span>
        <div class="code-box">${escapeHtml(cp.mitigation_cmd)}</div>
        <span style="font-size: 11px; font-weight: 600; color: #00d2ff; margin-top: 8px; display: inline-block;">BloodHound Cypher:</span>
        <div class="code-box">${escapeHtml(cp.cypher_query)}</div>
      `;
    }

    window.resetAllSeverances = function() {
      severedEdges.forEach(eid => {
        const edge = cy.getElementById(eid);
        edge.removeClass('severed');
        const btn = document.getElementById(`btn-${eid}`);
        if (btn) {
          btn.textContent = "✂ Sever Edge (Simulation)";
          btn.className = "sever-btn";
        }
        const card = document.getElementById(`cp-card-${eid}`);
        if (card) card.classList.remove("severed");
      });
      severedEdges.clear();
      updateReachabilityMeter(baselineProb);
    };

    function escapeHtml(str) {
      if (!str) return "";
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Auto-select first chokepoint
    if (PAYLOAD.chokepoints.length > 0) {
      showMitigationPlaybook(PAYLOAD.chokepoints[0]);
    }
  </script>
</body>
</html>
"""


def generate_corridor_visualization(
    harden_result: HardenResult,
    graph: Optional[nx.DiGraph] = None,
    output_path: str = "corridor.html",
    title: Optional[str] = None,
) -> str:
    """
    Generate an interactive, standalone HTML/JS visualization of the lateral movement corridor
    and critical chokepoints using Cytoscape.js.

    Args:
        harden_result: HardenResult from engine.harden().
        graph: Optional NetworkX DiGraph (if provided, extracts corridor context).
        output_path: Path to write the output HTML file.
        title: Optional custom report title.

    Returns:
        output_path: Path where the HTML was saved.
    """
    src_node = harden_result.source
    dst_node = harden_result.target
    bridge_node = harden_result.bridge_node
    path_nodes = set(harden_result.attack_path)

    title_str = title or f"NetReach Lateral Movement Corridor: {src_node} -> {dst_node}"

    # Build Cytoscape elements (nodes & edges)
    elements = []
    seen_nodes = set()
    node_degrees = {}

    # Map chokepoints by (source, target)
    chokepoint_map = {}
    chokepoint_list = []
    for cp in harden_result.chokepoints:
        cp_key = (cp.source, cp.target)
        edge_id = f"e_{cp.source}_to_{cp.target}".replace("@", "_").replace(".", "_").replace(":", "_").replace("/", "_")
        chokepoint_map[cp_key] = (cp, edge_id)
        chokepoint_list.append({
            "edge_id": edge_id,
            "rank": cp.rank,
            "source": cp.source,
            "target": cp.target,
            "relation": cp.relation,
            "ad_relation": cp.ad_relation,
            "baseline_prob": cp.baseline_prob,
            "ablated_prob": cp.ablated_prob,
            "delta_p": cp.delta_p,
            "mitigation_cmd": cp.mitigation_cmd,
            "cypher_query": cp.cypher_query,
            "description": cp.description,
        })

    # 1. Add attack path nodes first
    for n in harden_result.attack_path:
        if n in seen_nodes:
            continue
        seen_nodes.add(n)

        if n == src_node:
            role = "source"
            color = "#ef4444"
            size = 48
        elif n == dst_node:
            role = "target"
            color = "#f59e0b"
            size = 48
        elif n == bridge_node:
            role = "bridge"
            color = "#00d2ff"
            size = 46
        else:
            role = "path"
            color = "#8b5cf6"
            size = 36

        elements.append({
            "data": {
                "id": n,
                "label": n.split("@")[0].split(".")[0],
                "full_name": n,
                "role": role,
                "color": color,
                "size": size,
            }
        })

    # 2. Add edges along attack path
    path_hops = []
    for rel_edge in harden_result.path_relations:
        u = rel_edge["source"]
        v = rel_edge["target"]
        c_rel = rel_edge.get("relation", "AdminTo")
        ad_rel = rel_edge.get("ad_relation", c_rel)
        cp_info = chokepoint_map.get((u, v))

        edge_id = cp_info[1] if cp_info else f"e_{u}_to_{v}".replace("@", "_").replace(".", "_").replace(":", "_").replace("/", "_")
        is_cp = 1 if cp_info else 0
        edge_color = "#f59e0b" if is_cp else "#8b5cf6"
        edge_width = 3 if is_cp else 2

        elements.append({
            "data": {
                "id": edge_id,
                "source": u,
                "target": v,
                "relation": ad_rel,
                "is_chokepoint": is_cp,
                "color": edge_color,
                "width": edge_width,
            }
        })

        path_hops.append({
            "source": u,
            "target": v,
            "relation": ad_rel,
            "edge_id": edge_id,
        })

    # 3. Add context corridor edges & nodes from graph if available
    if graph is not None:
        for u, v, data in graph.edges(data=True):
            if u not in seen_nodes:
                seen_nodes.add(u)
                elements.append({
                    "data": {
                        "id": u,
                        "label": str(u).split("@")[0].split(".")[0],
                        "full_name": str(u),
                        "role": "corridor",
                        "color": "#475569",
                        "size": 28,
                    }
                })
            if v not in seen_nodes:
                seen_nodes.add(v)
                elements.append({
                    "data": {
                        "id": v,
                        "label": str(v).split("@")[0].split(".")[0],
                        "full_name": str(v),
                        "role": "corridor",
                        "color": "#475569",
                        "size": 28,
                    }
                })

            edge_key = (u, v)
            if edge_key not in [(h["source"], h["target"]) for h in path_hops]:
                rel = data.get("ad_relation") or data.get("cloud_relation") or data.get("relation", "Route")
                edge_id = f"e_ctx_{u}_to_{v}".replace("@", "_").replace(".", "_").replace(":", "_").replace("/", "_")
                elements.append({
                    "data": {
                        "id": edge_id,
                        "source": u,
                        "target": v,
                        "relation": rel,
                        "is_chokepoint": 0,
                        "color": "#334155",
                        "width": 1,
                    }
                })

    payload = {
        "telemetry": {
            "source": src_node,
            "target": dst_node,
            "baseline_probability": harden_result.baseline_probability,
            "risk_level": harden_result.baseline_risk_level,
            "act_steps": harden_result.act_steps,
            "bridge_node": bridge_node,
            "analysis_latency_ms": harden_result.analysis_latency_ms,
            "subgraph_nodes": harden_result.subgraph_nodes,
            "subgraph_edges": harden_result.subgraph_edges,
        },
        "chokepoints": chokepoint_list,
        "path_hops": path_hops,
        "elements": elements,
    }

    payload_json = json.dumps(payload)
    html_content = HTML_TEMPLATE.replace("__TITLE__", title_str).replace("__PAYLOAD_JSON__", payload_json)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
