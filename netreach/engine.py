"""
High-Level Inference Engine for NetReach.

Provides zero-dependency neural reachability auditing and latency benchmarking
with automatic CUDA acceleration and graceful CPU fallback.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import torch
import torch.nn.functional as F

from netreach.model import ChallengerLatentNet
from netreach.assets import DEFAULT_WEIGHTS_PATH
from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch


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
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_compile: bool = False,
    ):
        # 1. Device selection with graceful fallback
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 2. Instantiate 501K-parameter ChallengerLatentNet
        self.model = ChallengerLatentNet(num_nodes=50, train_steps=5, use_compile=use_compile).to(self.device)

        # 3. Load pre-trained weights
        weights_file = checkpoint_path or DEFAULT_WEIGHTS_PATH
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Checkpoint weights not found at: {weights_file}")

        ckpt = torch.load(weights_file, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.param_count = self.model.count_parameters()
        self.use_bf16 = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())

    def audit(
        self,
        graph_source: Union[str, nx.DiGraph, List[Dict[str, Any]], Dict[str, Any]],
        source: str,
        target: str,
        depths: Tuple[int, ...] = (5, 10, 15, 20),
        max_nodes: int = 48,
        max_edges: int = 96,
    ) -> AuditResult:
        """
        Audit lateral movement path reachability between source and target.

        Args:
            graph_source: File path (JSON, GraphML, DOT) or NetworkX DiGraph.
            source: Source node identifier (e.g., 'Workstation-01').
            target: Target node identifier (e.g., 'Domain-Controller-01').
            depths: Recurrent step counts T to evaluate test-time scaling.
            max_nodes: Maximum nodes retained in extracted corridor.
            max_edges: Maximum edges retained in extracted corridor.

        Returns:
            AuditResult containing reachability probabilities, risk tier, and diagnostics.
        """
        G = load_graph(graph_source)
        subG, meta = extract_subgraph(G, source=source, target=target, max_nodes=max_nodes, max_edges=max_edges)
        batch = prepare_graph_batch(subG, source=source, target=target, device=self.device, target_nodes=max_nodes, target_edges=max_edges)

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
                    logit, latents = self.model(batch, steps=step, return_latents=True)
                    prob = torch.sigmoid(logit).item()
                    cos_sim = compute_pairwise_cosine_similarity(latents)
                    probabilities[step] = prob
                    cosine_sims[step] = cos_sim
                    max_cos_sim = max(max_cos_sim, cos_sim)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        # Assess risk tier from final scaled probability and ground truth
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
        )

    def benchmark(
        self,
        num_samples: int = 100,
        hops: Tuple[int, ...] = (3, 5, 8, 12, 16),
        depths: Tuple[int, ...] = (5, 10, 15, 20),
    ) -> Dict[str, Any]:
        """
        Run an automated latency, parameter audit, and accuracy benchmark.
        """
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
                autocast_ctx = (
                    torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.use_bf16
                    else torch.amp.autocast(device_type="cpu", enabled=False)
                )
                with autocast_ctx:
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
