"""
NetReach v3: Bidirectional Wavefront Recurrence (Meet-in-the-Middle) with Collision Detection,
Adaptive Computation Time (ACT), and Multi-Task Pointer Decoding.

Architectural Specs:
- Parameter budget: Strictly between 15.0M and 22.0M parameters.
- Hidden dimension: d_model = 512, FFN hidden dim = 1024, 8 attention heads (d_k = 64).
- Relation embedding dimension: 128 (supports 6 directed relation types + padding).
- 2-layer stacked relational message-passing blocks per unroll step.
- Dual-wavefront recurrence: forward wave along outgoing edges, backward wave along incoming edges.
- Collision Detection: C_t(v) = (W_f H^{(fwd)}_v)^T (W_b H^{(bwd)}_v) / sqrt(d).
- ACT halting conditioned on [max_v C_t(v), mean_v C_t(v)].
- Multi-Task Heads:
  1. Reachability logit (BCEWithLogitsLoss).
  2. Hop count regression (SmoothL1Loss).
  3. Pointer Decoder (CrossEntropyLoss over next-hop successors).
"""

from typing import Dict, List, Optional, Tuple, Union
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


class RelationalMessagePassingLayer(nn.Module):
    """
    Single relational message-passing layer with multi-head relation-biased attention,
    vectorized max aggregation, and GRU-style highway gating with persistent source anchor.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 1024,
        nhead: int = 8,
        rel_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.rel_dim = rel_dim

        # Multi-head attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(rel_dim, d_model)
        self.W_vr = nn.Linear(rel_dim, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_norm = RMSNorm(d_model)

        # Vectorized max edge aggregation
        self.max_msg_proj = nn.Linear(d_model, d_model)

        # Highway Gating with Persistent Anchor
        in_dim = d_model * 4  # [H, attn_msg, max_msg, H_0]
        self.gate_linear = nn.Linear(in_dim, d_model)
        self.cand_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute one relational message-passing step.

        Args:
            H: Current node latents [B, N, d].
            H_0: Persistent source anchor latents [B, N, d].
            edge_src: Source node indices of directed transmission [B, M].
            edge_dst: Destination node indices of directed transmission [B, M].
            edge_rel_embed: Pre-computed edge relation embeddings [B, M, rel_dim].

        Returns:
            Updated node latents H_next [B, N, d].
        """
        B, N, d = H.shape
        M_edges = edge_src.shape[1]

        # 1. Multi-head relational attention
        Q = self.W_q(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)

        W_r_R = self.W_r(edge_rel_embed).view(B, M_edges, self.nhead, self.d_k).permute(0, 2, 1, 3)
        W_vr_R = self.W_vr(edge_rel_embed).view(B, M_edges, self.nhead, self.d_k).permute(0, 2, 1, 3)

        idx_dst = edge_dst.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)
        idx_src = edge_src.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)

        Q_dst = torch.gather(Q, 2, idx_dst)
        K_src = torch.gather(K, 2, idx_src)
        V_src = torch.gather(V, 2, idx_src) + W_vr_R

        # Integrate learned relation embedding R into attention score
        K_rel = K_src + W_r_R
        scores = (Q_dst * K_rel).sum(dim=-1) / (self.d_k ** 0.5)  # [B, heads, M]

        # Scatter attention scores into destination matrix S [B, heads, N, N]
        S = torch.full((B, self.nhead, N, N), -1e9, device=H.device, dtype=scores.dtype)
        b_idx = torch.arange(B, device=H.device).view(B, 1, 1).expand(B, self.nhead, M_edges)
        h_idx = torch.arange(self.nhead, device=H.device).view(1, self.nhead, 1).expand(B, self.nhead, M_edges)
        v_idx = edge_dst.unsqueeze(1).expand(B, self.nhead, M_edges)
        u_idx = edge_src.unsqueeze(1).expand(B, self.nhead, M_edges)

        S[b_idx, h_idx, v_idx, u_idx] = scores
        attn_weights = F.softmax(S, dim=-1)
        has_in = (S > -1e4).any(dim=-1, keepdim=True)
        attn_weights = torch.where(has_in, attn_weights, torch.zeros_like(attn_weights))

        attn_msg = torch.matmul(attn_weights, V)
        attn_msg = attn_msg.permute(0, 2, 1, 3).contiguous().view(B, N, d)
        attn_msg = self.attn_norm(self.W_o(attn_msg))

        # 2. Vectorized Max Aggregation over edge buffers
        M_proj = self.max_msg_proj(H)
        edge_messages = torch.gather(M_proj, 1, edge_src.unsqueeze(-1).expand(-1, -1, d))
        dst_index = edge_dst.unsqueeze(-1).expand(-1, -1, d)
        max_msg = torch.zeros(B, N, d, device=M_proj.device, dtype=M_proj.dtype)
        max_msg = max_msg.scatter_reduce(1, dst_index, edge_messages, reduce="amax", include_self=False)

        # 3. Gated Recurrent Update with Persistent Source Anchor H_0
        combined = torch.cat([H, attn_msg, max_msg, H_0], dim=-1)
        z = torch.sigmoid(self.gate_linear(combined))
        H_tilde = self.cand_mlp(combined)
        H_next = (1.0 - z) * H + z * H_tilde
        H_next = self.norm(H_next)

        active = (
            (H_0.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (H.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (attn_msg.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (max_msg.abs().sum(dim=-1, keepdim=True) > 1e-4)
        )
        return H_next * active.float()


class ChallengerLatentNetV3(nn.Module):
    """
    NetReach v3 Dual-Wavefront Recurrent Latent Network (Meet-in-the-Middle).

    Propagates forward and backward waves simultaneously, evaluating collision affinity
    C_t(v) to detect bridge nodes in ~ ceil(H / 2) steps.
    """

    def __init__(
        self,
        num_nodes: int = 128,
        node_embed_dim: int = 256,
        d_model: int = 512,
        hidden_dim: int = 1024,
        nhead: int = 8,
        rel_dim: int = 128,
        num_relations: int = 8,
        max_steps: int = 14,
        epsilon: float = 0.01,
        use_compile: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_dim = node_embed_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.rel_dim = rel_dim
        self.num_relations = num_relations
        self.max_steps = max_steps
        self.epsilon = epsilon

        # Node identity and initial projections
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)
        self.node_proj = nn.Linear(node_embed_dim + 3, d_model)
        self.rel_embed = nn.Embedding(num_relations, rel_dim)

        # 2-layer stacked relational message-passing blocks
        self.layer1 = RelationalMessagePassingLayer(d_model, hidden_dim, nhead, rel_dim)
        self.layer2 = RelationalMessagePassingLayer(d_model, hidden_dim, nhead, rel_dim)

        # Collision detection projections
        # C_t(v) = (W_f H_v^(fwd))^T (W_b H_v^(bwd)) / sqrt(d)
        self.W_f = nn.Linear(d_model, d_model)
        self.W_b = nn.Linear(d_model, d_model)

        # ACT Halting head on collision statistics [max_v C_t(v), mean_v C_t(v)]
        self.halt_head = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Multi-task heads
        # Pooled representation: [h_dst^(fwd), h_src^(bwd), h_bridge, h_src^(fwd) * h_dst^(bwd), max_c, mean_c]
        query_dim = d_model * 4 + 2
        self.reach_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hop_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Head 3: Pointer Decoder for predicting next-hop successor along the path
        self.pointer_q = nn.Linear(d_model * 2, d_model)
        self.pointer_k = nn.Linear(d_model * 2, d_model)

        total_params = self.count_parameters()
        print(f"[Init] ChallengerLatentNetV3 initialized with {total_params:,} parameters.")
        assert (
            15_000_000 <= total_params <= 22_000_000
        ), f"Parameter count {total_params:,} violates [15.0M, 22.0M] budget constraint!"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _unroll_stacked_step(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Run 2-layer stacked message-passing block."""
        H_mid = self.layer1(H, H_0, edge_src, edge_dst, edge_rel_embed)
        return self.layer2(H_mid, H_0, edge_src, edge_dst, edge_rel_embed)

    def _compute_collision(
        self,
        H_fwd: torch.Tensor,
        H_bwd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute node-level collision affinity:
        C_t(v) = (W_f H^{(fwd)}_v)^T (W_b H^{(bwd)}_v) / sqrt(d)
        """
        Z_f = self.W_f(H_fwd)  # [B, N, d]
        Z_b = self.W_b(H_bwd)  # [B, N, d]
        raw_C_t = (Z_f * Z_b).sum(dim=-1) / (self.d_model ** 0.5)  # [B, N]

        # Only nodes reached by BOTH waves can produce valid collision
        active_both = (H_fwd.abs().sum(dim=-1) > 1e-4) & (H_bwd.abs().sum(dim=-1) > 1e-4)
        C_t = torch.where(active_both, raw_C_t, torch.full_like(raw_C_t, -10.0))
        max_c = C_t.max(dim=-1)[0]  # [B]
        
        has_any = active_both.any(dim=-1)
        mean_c = torch.where(
            has_any,
            (torch.where(active_both, raw_C_t, torch.zeros_like(raw_C_t)).sum(dim=-1) / torch.clamp(active_both.sum(dim=-1).float(), min=1.0)),
            torch.tensor(-10.0, device=C_t.device)
        )
        return C_t, max_c, mean_c

    def _pool_predictions(
        self,
        H_fwd: torch.Tensor,
        H_bwd: torch.Tensor,
        C_t: torch.Tensor,
        max_c: torch.Tensor,
        mean_c: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce reachability logit, hop count prediction, and pointer successor logits.
        """
        B, N, d = H_fwd.shape
        b_range = torch.arange(B, device=H_fwd.device)

        h_src_fwd = H_fwd[b_range, src_idx]  # [B, d]
        h_dst_fwd = H_fwd[b_range, dst_idx]  # [B, d]
        h_src_bwd = H_bwd[b_range, src_idx]  # [B, d]
        h_dst_bwd = H_bwd[b_range, dst_idx]  # [B, d]

        # Soft bridge collision pooling across nodes
        beta = F.softmax(C_t, dim=-1).unsqueeze(-1)  # [B, N, 1]
        h_bridge = (beta * (H_fwd + H_bwd)).sum(dim=1)  # [B, d]

        # Cross-direction interaction
        h_cross = h_src_fwd * h_dst_bwd  # [B, d]
        c_stats = torch.stack([max_c, mean_c], dim=-1)  # [B, 2]

        h_query = torch.cat([h_dst_fwd, h_src_bwd, h_bridge, h_cross, c_stats], dim=-1)

        reach_logit = self.reach_head(h_query).squeeze(-1) + 2.0 * max_c
        hop_pred = self.hop_head(h_query).squeeze(-1)

        # Pointer logits: Q_u @ K_v^T / sqrt(d)
        H_both = torch.cat([H_fwd, H_bwd], dim=-1)  # [B, N, 2*d]
        Q_ptr = self.pointer_q(H_both)  # [B, N, d]
        K_ptr = self.pointer_k(H_both)  # [B, N, d]
        ptr_scores = torch.bmm(Q_ptr, K_ptr.transpose(1, 2)) / (self.d_model ** 0.5)  # [B, N, N]

        if adj is not None:
            # Mask out non-edges while allowing self-loops (end-of-path)
            I_mask = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            valid_edges = torch.clamp(adj + I_mask, max=1.0)
            ptr_scores = ptr_scores - 1e4 * (1.0 - valid_edges)

        return reach_logit, hop_pred, ptr_scores

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        return_latents: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual-wavefront recurrence and adaptive halting.

        Args:
            batch: Dict with node_ids, indicators, adj, edge_src, edge_dst, edge_rel, src_idx, dst_idx.
            steps: If specified, unroll for exactly `steps` fixed iterations.
                   If None, execute ACT halting up to max_steps.
            return_latents: If True, return final forward/backward latents and collision map.

        Returns:
            Dict with:
                'logit': reachability logit tensor [B],
                'hop': predicted hop count tensor [B],
                'pointer_logits': next-hop scores [B, N, N],
                'steps_taken': mean steps taken per sample [B],
                'ponder_cost': ACT ponder loss scalar,
                'collision_map': [B, N] (if return_latents).
        """
        node_ids = batch["node_ids"]
        indicators_fwd = batch["indicators"]
        adj = batch["adj"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_rel = batch.get("edge_rel")
        src_idx = batch["src_idx"]
        dst_idx = batch["dst_idx"]

        B, N = node_ids.shape
        M_edges = edge_src.shape[1]

        if edge_rel is None:
            edge_rel = torch.zeros(B, M_edges, dtype=torch.long, device=edge_src.device)

        # Pre-embed relation types
        edge_rel_embed = self.rel_embed(edge_rel)  # [B, M, rel_dim]

        # Construct backward indicators: dst is query start [1, 0, 0], src is query target [0, 1, 0]
        indicators_bwd = torch.zeros_like(indicators_fwd)
        indicators_bwd[:, :, 2] = 1.0  # background nodes
        b_range = torch.arange(B, device=node_ids.device)
        indicators_bwd[b_range, dst_idx] = torch.tensor([1.0, 0.0, 0.0], device=node_ids.device)
        indicators_bwd[b_range, src_idx] = torch.tensor([0.0, 1.0, 0.0], device=node_ids.device)

        # Node initial embeddings and query wavefront seeding
        node_indices = torch.arange(N, device=node_ids.device).unsqueeze(0).expand(B, N)
        is_src = (node_indices == src_idx.unsqueeze(1)).unsqueeze(-1).float()
        is_dst = (node_indices == dst_idx.unsqueeze(1)).unsqueeze(-1).float()

        node_repr = self.node_embed(node_ids)  # [B, N, node_embed_dim]
        # Only query source/destination nodes are seeded with initial wavefront energy
        H_0_fwd = self.node_proj(torch.cat([node_repr, indicators_fwd], dim=-1)) * is_src
        H_0_bwd = self.node_proj(torch.cat([node_repr, indicators_bwd], dim=-1)) * is_dst

        H_fwd = H_0_fwd.clone()
        H_bwd = H_0_bwd.clone()

        # Backward edge buffers: edges reversed (v -> u)
        edge_src_bwd = edge_dst
        edge_dst_bwd = edge_src

        # Fixed steps unrolling mode
        if steps is not None:
            for _ in range(steps):
                H_fwd = self._unroll_stacked_step(H_fwd, H_0_fwd, edge_src, edge_dst, edge_rel_embed)
                H_bwd = self._unroll_stacked_step(H_bwd, H_0_bwd, edge_src_bwd, edge_dst_bwd, edge_rel_embed)

            C_t, max_c, mean_c = self._compute_collision(H_fwd, H_bwd)
            logit, hop, ptr_scores = self._pool_predictions(
                H_fwd, H_bwd, C_t, max_c, mean_c, src_idx, dst_idx, adj
            )
            res = {
                "logit": logit,
                "hop": hop,
                "pointer_logits": ptr_scores,
                "steps_taken": torch.full((B,), float(steps), device=H_fwd.device),
                "ponder_cost": torch.tensor(0.0, device=H_fwd.device),
                "max_collision": max_c,
            }
            if return_latents:
                res["H_fwd"] = H_fwd
                res["H_bwd"] = H_bwd
                res["collision_map"] = C_t
            return res

        # Autonomous Adaptive Computation Time (ACT) mode
        halting_prob = torch.zeros(B, device=H_fwd.device)
        remainders = torch.zeros(B, device=H_fwd.device)
        n_updates = torch.zeros(B, device=H_fwd.device)
        steps_taken = torch.zeros(B, device=H_fwd.device)

        accum_H_fwd = torch.zeros_like(H_fwd)
        accum_H_bwd = torch.zeros_like(H_bwd)
        accum_C_t = torch.zeros(B, N, device=H_fwd.device)
        accum_max_c = torch.zeros(B, device=H_fwd.device)
        accum_mean_c = torch.zeros(B, device=H_fwd.device)

        still_running = torch.ones(B, dtype=torch.bool, device=H_fwd.device)
        last_C_t = torch.zeros(B, N, device=H_fwd.device)

        for t in range(1, self.max_steps + 1):
            H_fwd = self._unroll_stacked_step(H_fwd, H_0_fwd, edge_src, edge_dst, edge_rel_embed)
            H_bwd = self._unroll_stacked_step(H_bwd, H_0_bwd, edge_src_bwd, edge_dst_bwd, edge_rel_embed)

            C_t, max_c, mean_c = self._compute_collision(H_fwd, H_bwd)
            last_C_t = C_t

            # Halting probability conditioned on collision affinity
            c_input = torch.stack([max_c, mean_c], dim=-1)
            p_t = torch.sigmoid(self.halt_head(c_input)).squeeze(-1)  # [B]

            # ACT probability thresholding
            will_halt = (halting_prob + p_t >= 1.0 - self.epsilon) & still_running
            active = still_running & ~will_halt

            halting_prob = halting_prob + p_t * active.float()
            remainders = torch.where(will_halt, 1.0 - halting_prob, remainders)
            halting_prob = torch.where(will_halt, torch.ones_like(halting_prob), halting_prob)

            n_updates = n_updates + active.float() + will_halt.float()
            steps_taken = torch.where(still_running, torch.full_like(steps_taken, float(t)), steps_taken)

            # Weight contribution
            w_t = torch.where(will_halt, remainders, p_t * active.float())  # [B]
            w_node = w_t.view(B, 1, 1)
            accum_H_fwd = accum_H_fwd + w_node * H_fwd
            accum_H_bwd = accum_H_bwd + w_node * H_bwd
            accum_C_t = accum_C_t + w_t.view(B, 1) * C_t
            accum_max_c = accum_max_c + w_t * max_c
            accum_mean_c = accum_mean_c + w_t * mean_c

            still_running = still_running & ~will_halt
            if not still_running.any() and not self.training:
                break

        # Handle unfinished samples at max_steps
        if still_running.any():
            remainder_final = 1.0 - halting_prob
            accum_H_fwd = accum_H_fwd + remainder_final.view(B, 1, 1) * H_fwd
            accum_H_bwd = accum_H_bwd + remainder_final.view(B, 1, 1) * H_bwd
            accum_C_t = accum_C_t + remainder_final.view(B, 1) * C_t
            accum_max_c = accum_max_c + remainder_final * max_c
            accum_mean_c = accum_mean_c + remainder_final * mean_c

        ponder_cost = (n_updates + remainders).mean()

        logit, hop, ptr_scores = self._pool_predictions(
            accum_H_fwd, accum_H_bwd, accum_C_t, accum_max_c, accum_mean_c, src_idx, dst_idx, adj
        )

        res = {
            "logit": logit,
            "hop": hop,
            "pointer_logits": ptr_scores,
            "steps_taken": steps_taken,
            "ponder_cost": ponder_cost,
        }
        if return_latents:
            res["H_fwd"] = H_fwd
            res["H_bwd"] = H_bwd
            res["collision_map"] = last_C_t
        return res
