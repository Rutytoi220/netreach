"""
High-Level Inference Engine for NetReach v0.6.0.

Provides zero-dependency neural reachability auditing, neural attack path tracing,
automated chokepoint analysis & remediation, and latency benchmarking with automatic
CUDA acceleration, graceful CPU fallback, and optional INT8 dynamic quantization.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import torch
import torch.nn.functional as F

from netreach.model_v4 import ChallengerLatentNetV4
from netreach.model import ChallengerLatentNet
from netreach.assets import DEFAULT_WEIGHTS_PATH, DEFAULT_V4_WEIGHTS_PATH
from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch
from netreach.export import quantize_model_for_cpu


@dataclass
class AuditResult:
    """Structured result of a NetReach attack path audit."""
    source: str
    target: str
    probabilities: Dict[int, float]
    cosine_similarities: Dict[int, float]
    risk_level: str
    risk_color: str
    ground_truth_path: List[str]
    shortest_hop_count: Optional[int]
    inference_latency_ms: float
    subgraph_nodes: int
    subgraph_edges: int
    oversmoothed: bool
    # v0.5.0 Neural Path Tracing & Halting Telemetry
    bridge_node: Optional[str] = None
    neural_path: List[str] = field(default_factory=list)
    path_relations: List[Dict[str, str]] = field(default_factory=list)
    is_valid_walk: bool = True
    act_steps: float = 0.0
    peak_collision: float = 0.0


def compute_pairwise_cosine_similarity(H: torch.Tensor) -> float:
    """Computes average off-diagonal pairwise cosine similarity between node latents."""
    B, N, d = H.shape
    if N <= 1:
        return 0.0
    H_norm = F.normalize(H, p=2, dim=-1)
    sim_matrix = torch.bmm(H_norm, H_norm.transpose(1, 2))
    eye_mask = torch.eye(N, dtype=torch.bool, device=H.device).unsqueeze(0).expand(B, -1, -1)
    return sim_matrix[~eye_mask].mean().item()


class NetReachEngine:
    """
    Production-grade neural engine for attack path reachability and lateral movement auditing.
    Defaults to 38M-parameter ChallengerLatentNetV4 with momentum wavefronts and pointer beam search.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_version: str = "v4",
        use_int8: bool = False,
        use_compile: bool = False,
        num_nodes: int = 256,
    ):
        self.model_version = model_version.lower()
        self.use_int8 = use_int8

        # 1. Device selection with graceful fallback
        if device is None:
            if torch.cuda.is_available() and not use_int8:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.use_int8 and self.device.type != "cpu":
            # Dynamic quantization runs on CPU
            self.device = torch.device("cpu")

        # 2. Instantiate Architecture
        if self.model_version == "v1":
            self.model = ChallengerLatentNet(num_nodes=50, train_steps=5, use_compile=use_compile).to(self.device)
            weights_file = checkpoint_path or DEFAULT_WEIGHTS_PATH
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"Checkpoint weights not found at: {weights_file}")
            ckpt = torch.load(weights_file, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
        else:
            # v4: 38M params ChallengerLatentNetV4
            self.model = ChallengerLatentNetV4(
                num_nodes=num_nodes,
                node_embed_dim=256,
                d_model=640,
                hidden_dim=1280,
                nhead=8,
                rel_dim=128,
                num_relations=8,
                max_steps=32,
            ).to(self.device)

            # Locate checkpoint: user-specified -> DEFAULT_V4_WEIGHTS_PATH -> results/checkpoints
            weights_file = checkpoint_path
            if weights_file is None:
                if os.path.exists(DEFAULT_V4_WEIGHTS_PATH):
                    weights_file = DEFAULT_V4_WEIGHTS_PATH
                else:
                    alt_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "results", "checkpoints", "challengerlatentnetv4_best.pt"
                    )
                    if os.path.exists(alt_path):
                        weights_file = alt_path

            if weights_file and os.path.exists(weights_file):
                ckpt = torch.load(weights_file, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)

        # 3. Optional INT8 Dynamic Quantization for CPU
        if self.use_int8:
            self.model = quantize_model_for_cpu(self.model)

        self.model.eval()
        self.param_count = getattr(self.model, "count_parameters", lambda: sum(p.numel() for p in self.model.parameters()))()
        self.use_bf16 = (self.device.type == "cuda" and torch.cuda.is_bf16_supported() and not self.use_int8)

    def audit(
        self,
        graph_source: Union[str, nx.DiGraph, List[Dict[str, Any]], Dict[str, Any]],
        source: str,
        target: str,
        depths: Tuple[int, ...] = (8, 14, 20),
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> AuditResult:
        """
        Audit lateral movement path reachability between source and target with neural path tracing.

        Args:
            graph_source: File path (JSON, BloodHound, GraphML, DOT) or NetworkX DiGraph.
            source: Source node identifier (e.g., 'Workstation-01', 'USER@DOMAIN.LOCAL').
            target: Target node identifier (e.g., 'Domain-Controller-01', 'DC01.CORP.LOCAL').
            depths: Recurrent step counts T to evaluate test-time scaling.
            max_nodes: Maximum nodes retained in extracted corridor (default: 256 for v4, 48 for v1).
            max_edges: Maximum edges retained in extracted corridor (default: 768 for v4, 96 for v1).

        Returns:
            AuditResult containing reachability probabilities, risk tier, traced path, and diagnostics.
        """
        if max_nodes is None:
            max_nodes = 48 if self.model_version == "v1" else 256
        if max_edges is None:
            max_edges = 96 if self.model_version == "v1" else 768

        G = load_graph(graph_source)

        # Robust case-insensitive node identifier lookup for Active Directory entity names
        if source not in G:
            src_lower = str(source).lower()
            for n in G.nodes():
                if str(n).lower() == src_lower:
                    source = n
                    break
        if target not in G:
            dst_lower = str(target).lower()
            for n in G.nodes():
                if str(n).lower() == dst_lower:
                    target = n
                    break

        subG, meta = extract_subgraph(G, source=source, target=target, max_nodes=max_nodes, max_edges=max_edges)
        batch = prepare_graph_batch(
            subG, source=source, target=target, device=self.device, target_nodes=max_nodes, target_edges=max_edges
        )

        probabilities = {}
        cosine_sims = {}
        max_cos_sim = 0.0

        t0 = time.perf_counter()
        with torch.no_grad():
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.use_bf16
                else torch.amp.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                for step in depths:
                    if self.model_version == "v1":
                        logit, latents = self.model(batch, steps=step, return_latents=True)
                        prob = torch.sigmoid(logit).item()
                        cos_sim = compute_pairwise_cosine_similarity(latents)
                    else:
                        out = self.model(batch, steps=step, return_latents=True)
                        logit = out["logit"]
                        prob = torch.sigmoid(logit).item()
                        cos_sim = compute_pairwise_cosine_similarity(out["H_fwd"])

                    probabilities[step] = prob
                    cosine_sims[step] = cos_sim
                    max_cos_sim = max(max_cos_sim, cos_sim)

                # ACT Collision Peak Halting & Neural Path Tracing for v4
                bridge_node_name = None
                neural_path = []
                path_relations = []
                is_valid_walk = True
                act_steps = float(depths[-1])
                peak_collision = 0.0

                if self.model_version != "v1" and hasattr(self.model, "trace_attack_path"):
                    # Halting telemetry
                    halt_out = self.model(batch, steps=None)
                    act_steps = halt_out["steps_taken"].item() if "steps_taken" in halt_out else float(depths[-1])
                    if "max_collision" in halt_out:
                        peak_collision = halt_out["max_collision"].item()

                    # Neural path tracing from bridge node
                    node_names = batch.get("node_names")
                    trace_res = self.model.trace_attack_path(
                        batch, sample_idx=0, node_names=node_names, max_steps=32
                    )
                    bridge_node_name = trace_res.get("bridge_node_name")
                    neural_path = trace_res.get("path_names", [])
                    is_valid_walk = trace_res.get("is_valid_walk", True)

                    # Extract edge relation metadata along traced path
                    fullG = batch.get("fullG", subG)
                    for i in range(len(neural_path) - 1):
                        u_node = neural_path[i]
                        v_node = neural_path[i + 1]
                        edata = subG.get_edge_data(u_node, v_node) or fullG.get_edge_data(u_node, v_node) or {}
                        canon_rel = edata.get("relation", "AdminTo")
                        ad_rel = edata.get("ad_relation", canon_rel)
                        path_relations.append({
                            "source": u_node,
                            "target": v_node,
                            "relation": canon_rel,
                            "ad_relation": ad_rel,
                        })

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Assess risk tier
        final_prob = probabilities[depths[-1]]
        if final_prob >= 0.75:
            risk_level = "CRITICAL"
            risk_color = "red"
        elif final_prob >= 0.50:
            risk_level = "HIGH"
            risk_color = "yellow"
        elif final_prob >= 0.30:
            risk_level = "MEDIUM"
            risk_color = "cyan"
        else:
            risk_level = "LOW / UNREACHABLE"
            risk_color = "green"

        return AuditResult(
            source=source,
            target=target,
            probabilities=probabilities,
            cosine_similarities=cosine_sims,
            risk_level=risk_level,
            risk_color=risk_color,
            ground_truth_path=meta["shortest_path"],
            shortest_hop_count=meta["shortest_path_length"],
            inference_latency_ms=latency_ms,
            subgraph_nodes=meta["extracted_nodes"],
            subgraph_edges=meta["extracted_edges"],
            oversmoothed=(max_cos_sim >= 0.85),
            bridge_node=bridge_node_name,
            neural_path=neural_path,
            path_relations=path_relations,
            is_valid_walk=is_valid_walk,
            act_steps=act_steps,
            peak_collision=peak_collision,
        )

    def benchmark(
        self,
        num_samples: int = 50,
        hops: Tuple[int, ...] = (6, 12, 18, 24),
        depths: Tuple[int, ...] = (8, 14, 20),
    ) -> Dict[str, Any]:
        """
        Run an automated latency, parameter audit, and accuracy benchmark.
        """
        if self.model_version == "v1":
            from netreach.adapters.generator import generate_benchmark_sample, collate_benchmark_batch
            latencies = []
            hop_accuracies = {h: {d: 0 for d in depths} for h in hops}
            hop_counts = {h: 0 for h in hops}

            with torch.no_grad():
                for i in range(num_samples):
                    h = hops[i % len(hops)]
                    reachable = (i % 2 == 0)
                    edges, src, dst, label, sample_h = generate_benchmark_sample(H=h, reachable=reachable, num_nodes=48, target_edges=96)
                    batch = collate_benchmark_batch(edges, src, dst, label, num_nodes=48, target_edges=96, device=self.device)
                    label_val = batch["label"].item()

                    t0 = time.perf_counter()
                    for d in depths:
                        logit = self.model(batch, steps=d)
                        pred = 1.0 if torch.sigmoid(logit).item() >= 0.5 else 0.0
                        if pred == label_val:
                            hop_accuracies[h][d] += 1
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
                    hop_counts[h] += 1

            mean_latency_ms = sum(latencies) / len(latencies)
            accuracy_summary = {}
            for h in hops:
                cnt = max(hop_counts[h], 1)
                accuracy_summary[f"H={h}"] = {f"T={d}": round(hop_accuracies[h][d] / cnt, 4) for d in depths}

            return {
                "device": str(self.device),
                "parameter_count": self.param_count,
                "mean_latency_ms": round(mean_latency_ms, 3),
                "latency_p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 3),
                "accuracy_summary": accuracy_summary,
                "tested_samples": num_samples,
            }

        # v4 benchmarking
        from data.generator import create_dataloader

        latencies = []
        hop_accuracies = {h: {d: 0 for d in depths} for h in hops}
        hop_counts = {h: 0 for h in hops}

        with torch.no_grad():
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.use_bf16
                else torch.amp.autocast(device_type="cpu", enabled=False)
            )
            per_hop = max(num_samples // len(hops), 1)
            for h in hops:
                loader = create_dataloader(
                    hop_lengths=[h],
                    num_samples=per_hop,
                    batch_size=1,
                    num_nodes=256,
                    target_edges=768,
                    shuffle=False,
                    seed=42 + h,
                )
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    label_val = batch["label"].item()
                    t0 = time.perf_counter()
                    with autocast_ctx:
                        for d in depths:
                            out = self.model(batch, steps=d)
                            pred = 1.0 if torch.sigmoid(out["logit"]).item() >= 0.5 else 0.0
                            if pred == label_val:
                                hop_accuracies[h][d] += 1
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
                    hop_counts[h] += 1

        mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        accuracy_summary = {}
        for h in hops:
            cnt = max(hop_counts[h], 1)
            accuracy_summary[f"H={h}"] = {f"T={d}": round(hop_accuracies[h][d] / cnt, 4) for d in depths}

        return {
            "device": str(self.device),
            "parameter_count": self.param_count,
            "mean_latency_ms": round(mean_latency_ms, 3),
            "latency_p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 3) if latencies else 0.0,
            "accuracy_summary": accuracy_summary,
            "tested_samples": sum(hop_counts.values()),
        }

    def harden(
        self,
        graph_source: Union[str, nx.DiGraph, List[Dict[str, Any]], Dict[str, Any]],
        source: str,
        target: str,
        top_k: int = 3,
        steps: Optional[int] = None,
    ):
        """
        Identify critical attack corridor chokepoints and generate actionable remediations
        via fast neural edge ablation.

        Args:
            graph_source: Path to graph or networkx DiGraph.
            source: Attacker entry principal/node.
            target: High-value objective asset.
            top_k: Number of highest-impact chokepoints to return.
            steps: Recurrent evaluation steps T (defaults to ACT steps or 8).

        Returns:
            HardenResult with baseline telemetry, ranked chokepoint edges, and mitigation scripts.
        """
        from netreach.remediation import find_critical_chokepoints
        return find_critical_chokepoints(
            graph_source=graph_source,
            src=source,
            dst=target,
            top_k=top_k,
            engine=self,
            steps=steps,
        )
