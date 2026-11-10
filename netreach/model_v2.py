"""
NetReach v2: Scaled Multi-Relational Recurrent Latent Network with Adaptive Computation Time (ACT).

Architectural Highlights:
- Scaled Parameter Budget: Strictly between 4.5M and 5.5M parameters (exact count printed on init).
- Base dimension d = 384, FFN hidden dim = 768, 4 attention heads (d_k = 96).
- Multi-Relational Edge Modeling:
  Support 6 directed relation types:
  AdminTo, MemberOf, HasSession, AllowedToDelegate, OpenPort, SubnetRoute.
  Learned relation embeddings R_{uv} in R^{64} integrated into attention scores:
  alpha_{uv} = Softmax_u( (W_q H_v)^T (W_k H_u + W_r R_{uv}) / sqrt(d_k) )
- Autonomous Adaptive Halting (ACT):
  p_t = sigma(W_halt H_t) with cumulative probability accumulation until sum p_t >= 1.0 - epsilon
  or max steps T_max = 20.
- Multi-Task Prediction Heads:
  Head 1: Binary reachability logit (BCEWithLogitsLoss).
  Head 2: Shortest path hop count regression (SmoothL1Loss).
- Vectorized fused message passing and bfloat16 mixed precision support on CUDA and CPU.
"""

from typing import Dict, Optional, Tuple, Union, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

RELATION_TYPES = [
    "AdminTo",
    "MemberOf",
    "HasSession",
    "AllowedToDelegate",
    "OpenPort",
    "SubnetRoute",
]
RELATION_TO_ID = {name: i for i, name in enumerate(RELATION_TYPES)}
NUM_RELATIONS = len(RELATION_TYPES)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class ChallengerLatentNetV2(nn.Module):
    """
    NetReach v2 Multi-Relational Recurrent Latent Network with ACT Halting.
    Maintains per-node latent state H_t in R^{B x N x d}.
    """

    def __init__(
        self,
        num_nodes: int = 64,
        node_embed_dim: int = 128,
        d_model: int = 384,
        hidden_dim: int = 768,
        nhead: int = 4,
        rel_dim: int = 64,
        num_relations: int = 8,
        max_steps: int = 20,
        epsilon: float = 0.01,
        use_compile: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.rel_dim = rel_dim
        self.num_relations = num_relations
        self.max_steps = max_steps
        self.epsilon = epsilon

        # Node identity embeddings and initial projection
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)
        self.node_proj = nn.Linear(node_embed_dim + 3, d_model)

        # Multi-relational edge embeddings
        self.rel_embed = nn.Embedding(num_relations, rel_dim)

        # Multi-Relational Attention Weights
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(rel_dim, d_model)
        self.W_vr = nn.Linear(rel_dim, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_norm = RMSNorm(d_model)

        # Max Message Projection
        self.max_msg_proj = nn.Linear(d_model, d_model)

        # Gated Highway Recurrence with Persistent Source Anchor
        in_dim = d_model * 4  # [H_t, attn_msg, max_msg, H_0]
        self.gate_linear = nn.Linear(in_dim, d_model)
        self.cand_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = RMSNorm(d_model)

        # Autonomous Adaptive Halting (ACT) Head: p_t = sigma(W_halt H_t)
        self.halt_head = nn.Linear(d_model * 3, 1)

        # Multi-Task Prediction Heads
        # Head 1: Binary reachability classification
        self.reach_head = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Head 2: Shortest path hop distance regression
        self.hop_head = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        total_params = self.count_parameters()
        print(f"[Init] ChallengerLatentNetV2 initialized with {total_params:,} parameters.")
        assert (
            4_500_000 <= total_params <= 5_500_000
        ), f"Parameter count {total_params:,} violates [4.5M, 5.5M] constraint!"

        self._transition_fn = self._raw_transition
        if use_compile and hasattr(torch, "compile"):
            try:
                self._transition_fn = torch.compile(self._raw_transition, mode="reduce-overhead")
            except Exception:
                self._transition_fn = self._raw_transition

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _raw_transition(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        adj: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-relational transition step with attention and gated update:
        alpha_{uv} = Softmax_u( (W_q H_v)^T (W_k H_u + W_r R_{uv}) / sqrt(d_k) )
        """
        B, N, d = H.shape
        M_edges = edge_src.shape[1]

        # 1. Project Q, K, V for multi-head attention
        Q = self.W_q(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)  # [B, heads, N, d_k]
        K = self.W_k(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)  # [B, heads, N, d_k]
        V = self.W_v(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)  # [B, heads, N, d_k]

        # 2. Relation embedding projection for edges
        R = self.rel_embed(edge_rel)  # [B, M, rel_dim]
        W_r_R = self.W_r(R).view(B, M_edges, self.nhead, self.d_k).permute(0, 2, 1, 3)    # [B, heads, M, d_k]
        W_vr_R = self.W_vr(R).view(B, M_edges, self.nhead, self.d_k).permute(0, 2, 1, 3)  # [B, heads, M, d_k]

        # Gather destination and source vectors for each edge
        idx_dst = edge_dst.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)
        idx_src = edge_src.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)

        Q_dst = torch.gather(Q, 2, idx_dst)  # [B, heads, M, d_k]
        K_src = torch.gather(K, 2, idx_src)  # [B, heads, M, d_k]
        V_src = torch.gather(V, 2, idx_src)  # [B, heads, M, d_k]

        # Integrate learned relation embedding R_uv into attention score:
        # (W_q H_v)^T (W_k H_u + W_r R_uv) / sqrt(d_k)
        K_rel = K_src + W_r_R
        scores = (Q_dst * K_rel).sum(dim=-1) / (self.d_k ** 0.5)  # [B, heads, M]

        # Scatter attention scores into dense destination matrix S [B, heads, N, N]
        S = torch.full((B, self.nhead, N, N), -1e9, device=H.device, dtype=scores.dtype)
        b_idx = torch.arange(B, device=H.device).view(B, 1, 1).expand(B, self.nhead, M_edges)
        h_idx = torch.arange(self.nhead, device=H.device).view(1, self.nhead, 1).expand(B, self.nhead, M_edges)
        v_idx = edge_dst.unsqueeze(1).expand(B, self.nhead, M_edges)
        u_idx = edge_src.unsqueeze(1).expand(B, self.nhead, M_edges)

        S[b_idx, h_idx, v_idx, u_idx] = scores
        attn_weights = F.softmax(S, dim=-1)  # softmax along incoming nodes u (dim=-1)
        has_in = (S > -1e4).any(dim=-1, keepdim=True)
        attn_weights = torch.where(has_in, attn_weights, torch.zeros_like(attn_weights))

        # Attended incoming message
        attn_msg = torch.matmul(attn_weights, V)  # [B, heads, N, d_k]
        attn_msg = attn_msg.permute(0, 2, 1, 3).contiguous().view(B, N, d)
        attn_msg = self.attn_norm(self.W_o(attn_msg))

        # 3. Vectorized Max Aggregation over edge buffers
        M_proj = self.max_msg_proj(H)
        edge_messages = torch.gather(M_proj, 1, edge_src.unsqueeze(-1).expand(-1, -1, d))
        dst_index = edge_dst.unsqueeze(-1).expand(-1, -1, d)
        max_msg = torch.zeros(B, N, d, device=M_proj.device, dtype=M_proj.dtype)
        max_msg = max_msg.scatter_reduce(1, dst_index, edge_messages, reduce="amax", include_self=False)

        # 4. Gated Recurrent Update with Persistent Source Anchor H_0
        combined = torch.cat([H, attn_msg, max_msg, H_0], dim=-1)  # [B, N, 4*d]
        z = torch.sigmoid(self.gate_linear(combined))              # [B, N, d]
        H_tilde = self.cand_mlp(combined)                          # [B, N, d]
        H_next = (1.0 - z) * H + z * H_tilde
        return self.norm(H_next)

    def transition(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        adj: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return self._transition_fn(H, H_0, adj, edge_src, edge_dst, edge_rel)
        except Exception:
            self._transition_fn = self._raw_transition
            return self._raw_transition(H, H_0, adj, edge_src, edge_dst, edge_rel)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        return_latents: bool = False,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with support for both Autonomous Adaptive Halting (ACT) and fixed-step scaling.

        Args:
            batch: Dict with node_ids, indicators, adj, edge_src, edge_dst, edge_rel, src_idx, dst_idx.
            steps: If specified, unroll for exactly `steps` fixed iterations.
                   If None, execute Autonomous Adaptive Halting (ACT) up to max_steps.
            return_latents: If True, include final latent tensor H in output.

        Returns:
            Dict containing:
                'logit': reachability logit tensor [B],
                'hop': predicted hop count tensor [B],
                'ponder_cost': ACT ponder loss scalar (when steps is None),
                'steps_taken': mean steps taken per sample [B],
                'latents': (optional) final node latents [B, N, d].
        """
        node_ids = batch["node_ids"]
        indicators = batch["indicators"]
        adj = batch["adj"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_rel = batch.get("edge_rel")
        if edge_rel is None:
            edge_rel = torch.zeros_like(edge_src)
        src_idx = batch["src_idx"]
        dst_idx = batch["dst_idx"]

        B, N = node_ids.shape
        batch_idx = torch.arange(B, device=node_ids.device)

        # 1. Initial Node Projection & Persistent Source Anchor H_0
        emb = self.node_embed(node_ids)
        H_0 = self.norm(self.node_proj(torch.cat([emb, indicators], dim=-1)))
        H = H_0

        # Mode A: Fixed Steps Execution (for testing test-time compute scaling T in [5, 10, 15, 20])
        if steps is not None:
            for _ in range(steps):
                H = self.transition(H, H_0, adj, edge_src, edge_dst, edge_rel)
            h_src = H[batch_idx, src_idx]
            h_dst = H[batch_idx, dst_idx]
            h_query = torch.cat([h_src, h_dst, h_src * h_dst], dim=-1)

            logits = self.reach_head(h_query).squeeze(-1)
            hops = self.hop_head(h_query).squeeze(-1)

            if return_latents:
                return logits, hops, H
            return {
                "logit": logits,
                "hop": hops,
                "ponder_cost": torch.tensor(0.0, device=node_ids.device),
                "steps_taken": torch.full((B,), float(steps), device=node_ids.device),
            }

        # Mode B: Autonomous Adaptive Computation Time (ACT)
        halting_probs: List[torch.Tensor] = []
        step_logits: List[torch.Tensor] = []
        step_hops: List[torch.Tensor] = []

        cum_p = torch.zeros(B, device=node_ids.device)
        weights: List[torch.Tensor] = []

        for t in range(self.max_steps):
            H = self.transition(H, H_0, adj, edge_src, edge_dst, edge_rel)
            h_src = H[batch_idx, src_idx]
            h_dst = H[batch_idx, dst_idx]
            h_query = torch.cat([h_src, h_dst, h_src * h_dst], dim=-1)

            # ACT halting probability: p_t = sigma(W_halt H_t)
            p_t = torch.sigmoid(self.halt_head(h_query)).squeeze(-1)  # [B]
            halting_probs.append(p_t)

            # Predictions at current step
            l_t = self.reach_head(h_query).squeeze(-1)  # [B]
            hop_t = self.hop_head(h_query).squeeze(-1)  # [B]
            step_logits.append(l_t)
            step_hops.append(hop_t)

            # Compute remainder-bounded ACT weights
            still_running = (cum_p < 1.0 - self.epsilon).float()
            new_cum_p = cum_p + p_t
            exceeds = (new_cum_p >= 1.0 - self.epsilon).float()

            w = still_running * (exceeds * (1.0 - cum_p) + (1.0 - exceeds) * p_t)
            if t == self.max_steps - 1:
                w = torch.where(cum_p < 1.0 - self.epsilon, 1.0 - cum_p, w)
            weights.append(w)
            cum_p = cum_p + (w if t == self.max_steps - 1 else (still_running * p_t))

            # Early exit if all batch samples have halted (and at least 3 steps completed)
            if (cum_p >= 1.0 - self.epsilon).all() and t >= 3:
                break

        weights_stack = torch.stack(weights, dim=-1)      # [B, T_act]
        probs_stack = torch.stack(halting_probs, dim=-1)  # [B, T_act]
        logits_stack = torch.stack(step_logits, dim=-1)    # [B, T_act]
        hops_stack = torch.stack(step_hops, dim=-1)        # [B, T_act]

        final_logits = (weights_stack * logits_stack).sum(dim=-1)  # [B]
        final_hops = (weights_stack * hops_stack).sum(dim=-1)      # [B]

        # Ponder cost: sum(t * p_t)
        T_act = weights_stack.shape[-1]
        t_indices = torch.arange(1, T_act + 1, device=node_ids.device, dtype=torch.float32)
        ponder_cost = (probs_stack * t_indices).sum(dim=-1).mean()
        steps_taken = (weights_stack > 1e-4).float().sum(dim=-1)

        result: Dict[str, Any] = {
            "logit": final_logits,
            "hop": final_hops,
            "ponder_cost": ponder_cost,
            "steps_taken": steps_taken,
        }
        if return_latents:
            result["latents"] = H
        return result
