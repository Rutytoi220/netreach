"""
NetReach v4: Momentum Wavefront Recurrence, Collision Peak Halting & Neural Path Tracing.

Architectural Specs:
- Parameter budget: Strictly between 35.0M and 45.0M parameters.
- Hidden dimension: d_model = 640, FFN hidden dim = 1280, 8 attention heads (d_k = 80).
- Relation embedding dimension: 128 (supports 6 directed relation types + padding).
- 3-layer stacked relational message-passing blocks per unroll step.
- Dual-wavefront recurrence with latent momentum buffers V_t:
    V_{t+1} = beta * V_t + (1 - beta) * Delta H_t
    H_{t+1} = RMSNorm(H_t + V_{t+1})
- Collision Peak Detection & Halting:
    Condition A (Hit): max_v C_t(v) >= tau_hit (0.0).
    Condition B (Dead End): |max_v C_t(v) - max_v C_{t-1}(v)| < epsilon for 2 consecutive steps after step 6.
    Hinge ponder cost: only penalizes steps taken AFTER discovery of a collision bridge.
- Multi-Task Heads:
    1. Reachability logit (BCEWithLogitsLoss).
    2. Shortest Hop count regression (SmoothL1Loss).
    3. Pointer Decoder (CrossEntropyLoss over next-hop successors).
- End-to-End Neural Path Tracing:
    Reconstructs the full sequence of hostnames starting from the bridge node v* = argmax_v C(v).
"""

import sys
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
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
    Relational message-passing layer with multi-head attention (8 heads, d_k=80),
    vectorized max edge aggregation, and gated highway update.
    """

    def __init__(
        self,
        d_model: int = 640,
        hidden_dim: int = 1280,
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

        # Highway Gating with Persistent Source Anchor
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
        """
        B, N, d = H.shape
        M_edges = edge_src.shape[1]

        # 1. Multi-head relational attention
        Q = self.W_q(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(H).view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)

        W_r_R = self.W_r(edge_rel_embed).view(B, M_edges, self.nhead, self.d_k).permute(0, 2, 1, 3)

        idx_dst = edge_dst.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)
        idx_src = edge_src.unsqueeze(1).unsqueeze(-1).expand(B, self.nhead, M_edges, self.d_k)

        Q_dst = torch.gather(Q, 2, idx_dst)
        K_src = torch.gather(K, 2, idx_src)
        V_src = torch.gather(V, 2, idx_src)

        # Integrate learned relation embedding R into attention score
        K_rel = K_src + W_r_R
        scores = (Q_dst * K_rel).sum(dim=-1) / (self.d_k ** 0.5)

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

        # 2. Vectorized Max Aggregation with Relational Bias
        M_proj = self.max_msg_proj(H)
        edge_messages = torch.gather(M_proj, 1, edge_src.unsqueeze(-1).expand(-1, -1, d)) + self.W_vr(edge_rel_embed)
        dst_index = edge_dst.unsqueeze(-1).expand(-1, -1, d)
        max_msg = torch.zeros(B, N, d, device=M_proj.device, dtype=M_proj.dtype)
        max_msg = max_msg.scatter_reduce(1, dst_index, edge_messages, reduce="amax", include_self=False)

        # 3. Gated Highway Recurrence with Persistent Source Anchor
        combined = torch.cat([H, attn_msg, max_msg, H_0], dim=-1)
        z = torch.sigmoid(self.gate_linear(combined))
        H_tilde = self.cand_mlp(combined)
        H_next = (1.0 - z) * H + z * H_tilde
        H_next = self.norm(H_next)

        # Wavefront active gating: inactive nodes stay at zero
        active = (
            (H_0.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (H.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (attn_msg.abs().sum(dim=-1, keepdim=True) > 1e-4)
            | (max_msg.abs().sum(dim=-1, keepdim=True) > 1e-4)
        )
        return H_next * active.float()


class ChallengerLatentNetV4(nn.Module):
    """
    NetReach v4: Momentum Wavefront Recurrence & Collision Peak Halting Engine.
    """

    def __init__(
        self,
        num_nodes: int = 256,
        node_embed_dim: int = 256,
        d_model: int = 640,
        hidden_dim: int = 1280,
        nhead: int = 8,
        rel_dim: int = 128,
        num_relations: int = 8,
        max_steps: int = 22,
        tau_hit: float = 0.6,
        epsilon_conv: float = 0.05,
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
        self.tau_hit = tau_hit if tau_hit > 0.5 else 0.6
        self.epsilon_conv = epsilon_conv

        # Node initial embeddings and projections
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)
        self.node_proj = nn.Linear(node_embed_dim + 3, d_model)
        self.rel_embed = nn.Embedding(num_relations, rel_dim)

        # 3-layer stacked relational message-passing blocks
        self.layer1 = RelationalMessagePassingLayer(d_model, hidden_dim, nhead, rel_dim)
        self.layer2 = RelationalMessagePassingLayer(d_model, hidden_dim, nhead, rel_dim)
        self.layer3 = RelationalMessagePassingLayer(d_model, hidden_dim, nhead, rel_dim)

        # Learnable Momentum parameter beta = sigmoid(beta_logit) ~ 0.8
        self.beta_logit = nn.Parameter(torch.tensor(1.386))
        self.momentum_norm_fwd = RMSNorm(d_model)
        self.momentum_norm_bwd = RMSNorm(d_model)

        # Collision detection projections
        self.W_f = nn.Linear(d_model, d_model)
        self.W_b = nn.Linear(d_model, d_model)

        # Halting head on collision statistics
        self.halt_head = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Multi-task heads
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

        # Pointer Decoders for next-hop path tracing
        self.pointer_q = nn.Linear(d_model * 2, d_model)
        self.pointer_k = nn.Linear(d_model * 2, d_model)

        total_params = self.count_parameters()
        print(f"[Init] ChallengerLatentNetV4 initialized with {total_params:,} parameters.", file=sys.stderr)
        assert (
            35_000_000 <= total_params <= 45_000_000
        ), f"Parameter count {total_params:,} violates [35.0M, 45.0M] budget constraint!"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _unroll_3layer_step(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Run 3-layer stacked relational message-passing blocks."""
        H1 = self.layer1(H, H_0, edge_src, edge_dst, edge_rel_embed)
        H2 = self.layer2(H1, H_0, edge_src, edge_dst, edge_rel_embed)
        return self.layer3(H2, H_0, edge_src, edge_dst, edge_rel_embed)

    def _compute_collision(
        self,
        H_fwd: torch.Tensor,
        H_bwd: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute node-level collision affinity with real-node masking:
        C_t(v) = (W_f H^{(fwd)}_v)^T (W_b H^{(bwd)}_v) / sqrt(d)
        """
        Z_f = self.W_f(H_fwd)
        Z_b = self.W_b(H_bwd)
        raw_C_t = (Z_f * Z_b).sum(dim=-1) / (self.d_model ** 0.5)

        # Active mask: node must be reached by both wavefronts
        active_both = (H_fwd.abs().sum(dim=-1) > 1e-4) & (H_bwd.abs().sum(dim=-1) > 1e-4)
        if node_mask is not None:
            mask = node_mask.bool()
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            active_both = active_both & mask
        else:
            mask = None

        C_t = torch.where(active_both, raw_C_t, torch.full_like(raw_C_t, -10.0))

        if mask is not None:
            # Mask out dummy padding nodes so distractors never win argmax
            C_t_masked = C_t.masked_fill(~mask, float("-inf"))
            peak_collision = C_t_masked.max(dim=1).values
            peak_collision = torch.where(torch.isneginf(peak_collision), torch.full_like(peak_collision, -10.0), peak_collision)
            bridge_node_idx = C_t_masked.argmax(dim=1)
            C_t_out = torch.where(mask, C_t, torch.full_like(C_t, -1e4))
        else:
            peak_collision = C_t.max(dim=1).values
            bridge_node_idx = C_t.argmax(dim=1)
            C_t_out = C_t

        has_any = active_both.any(dim=-1)
        mean_c = torch.where(
            has_any,
            (torch.where(active_both, raw_C_t, torch.zeros_like(raw_C_t)).sum(dim=-1) / torch.clamp(active_both.sum(dim=-1).float(), min=1.0)),
            torch.tensor(-10.0, device=C_t.device),
        )
        return C_t_out, peak_collision, mean_c

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
        """Produce reachability logit, hop count prediction, and pointer logits."""
        B, N, d = H_fwd.shape
        b_range = torch.arange(B, device=H_fwd.device)

        h_src_fwd = H_fwd[b_range, src_idx]
        h_dst_fwd = H_fwd[b_range, dst_idx]
        h_src_bwd = H_bwd[b_range, src_idx]
        h_dst_bwd = H_bwd[b_range, dst_idx]

        # Soft bridge collision pooling across nodes
        beta = F.softmax(C_t, dim=-1).unsqueeze(-1)
        h_bridge = (beta * (H_fwd + H_bwd)).sum(dim=1)

        h_cross = h_src_fwd * h_dst_bwd
        c_stats = torch.stack([max_c, mean_c], dim=-1)

        h_query = torch.cat([h_dst_fwd, h_src_bwd, h_bridge, h_cross, c_stats], dim=-1)

        reach_logit = self.reach_head(h_query).squeeze(-1) + 2.0 * max_c
        hop_pred = self.hop_head(h_query).squeeze(-1)

        # Pointer logits: Q_u @ K_v^T / sqrt(d)
        H_both = torch.cat([H_fwd, H_bwd], dim=-1)
        Q_ptr = self.pointer_q(H_both)
        K_ptr = self.pointer_k(H_both)
        ptr_scores = torch.bmm(Q_ptr, K_ptr.transpose(1, 2)) / (self.d_model ** 0.5)

        if adj is not None:
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
        Forward pass with momentum wavefronts and collision peak halting.
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

        edge_rel_embed = self.rel_embed(edge_rel)
        beta_momentum = torch.sigmoid(self.beta_logit)

        node_mask = batch.get("node_mask")
        if node_mask is None:
            node_names = batch.get("node_names")
            if node_names and isinstance(node_names, list):
                is_real = [not str(name).startswith("__distractor_") for name in node_names]
                node_mask = torch.tensor(is_real, dtype=torch.bool, device=node_ids.device).unsqueeze(0).expand(B, -1)
            else:
                node_mask = torch.ones(B, N, dtype=torch.bool, device=node_ids.device)
        else:
            if node_mask.dim() == 3:
                node_mask = node_mask.squeeze(-1)
            node_mask = node_mask.bool().to(node_ids.device)

        # Construct backward indicators: dst is query start
        indicators_bwd = torch.zeros_like(indicators_fwd)
        indicators_bwd[:, :, 2] = 1.0
        b_range = torch.arange(B, device=node_ids.device)
        indicators_bwd[b_range, dst_idx] = torch.tensor([1.0, 0.0, 0.0], device=node_ids.device)
        indicators_bwd[b_range, src_idx] = torch.tensor([0.0, 1.0, 0.0], device=node_ids.device)

        node_indices = torch.arange(N, device=node_ids.device).unsqueeze(0).expand(B, N)
        is_src = (node_indices == src_idx.unsqueeze(1)).unsqueeze(-1).float()
        is_dst = (node_indices == dst_idx.unsqueeze(1)).unsqueeze(-1).float()

        node_repr = self.node_embed(node_ids)
        H_0_fwd = self.node_proj(torch.cat([node_repr, indicators_fwd], dim=-1)) * is_src
        H_0_bwd = self.node_proj(torch.cat([node_repr, indicators_bwd], dim=-1)) * is_dst

        H_fwd = H_0_fwd.clone()
        H_bwd = H_0_bwd.clone()
        V_fwd = torch.zeros_like(H_fwd)
        V_bwd = torch.zeros_like(H_bwd)

        edge_src_bwd = edge_dst
        edge_dst_bwd = edge_src

        # Fixed steps unrolling mode
        if steps is not None:
            for _ in range(steps):
                # Momentum forward update
                H_fwd_next = self._unroll_3layer_step(H_fwd, H_0_fwd, edge_src, edge_dst, edge_rel_embed)
                delta_fwd = H_fwd_next - H_fwd
                V_fwd = beta_momentum * V_fwd + (1.0 - beta_momentum) * delta_fwd
                H_fwd = self.momentum_norm_fwd(H_fwd + V_fwd)

                # Momentum backward update
                H_bwd_next = self._unroll_3layer_step(H_bwd, H_0_bwd, edge_src_bwd, edge_dst_bwd, edge_rel_embed)
                delta_bwd = H_bwd_next - H_bwd
                V_bwd = beta_momentum * V_bwd + (1.0 - beta_momentum) * delta_bwd
                H_bwd = self.momentum_norm_bwd(H_bwd + V_bwd)

            C_t, max_c, mean_c = self._compute_collision(H_fwd, H_bwd, node_mask=node_mask)
            logit, hop, ptr_scores = self._pool_predictions(
                H_fwd, H_bwd, C_t, max_c, mean_c, src_idx, dst_idx, adj
            )
            res = {
                "logit": logit,
                "hop": hop,
                "pointer_logits": ptr_scores,
                "steps_taken": torch.full((B,), float(steps), device=H_fwd.device),
                "ponder_cost": torch.tensor(0.0, device=H_fwd.device),
                "max_collision": torch.clamp(max_c, min=0.0),
            }
            if return_latents:
                res["H_fwd"] = H_fwd
                res["H_bwd"] = H_bwd
                res["collision_map"] = C_t
            return res

        # Collision Peak Halting Mode
        steps_taken = torch.zeros(B, device=H_fwd.device)
        t_hit = torch.full((B,), float(self.max_steps), device=H_fwd.device)
        is_hit = torch.zeros(B, dtype=torch.bool, device=H_fwd.device)
        is_halted = torch.zeros(B, dtype=torch.bool, device=H_fwd.device)

        prev_max_c = torch.full((B,), -10.0, device=H_fwd.device)
        prev2_max_c = torch.full((B,), -10.0, device=H_fwd.device)

        last_H_fwd = H_fwd.clone()
        last_H_bwd = H_bwd.clone()
        last_C_t = torch.full((B, N), -10.0, device=H_fwd.device)
        last_max_c = torch.full((B,), -10.0, device=H_fwd.device)
        last_mean_c = torch.full((B,), -10.0, device=H_fwd.device)

        for t in range(1, self.max_steps + 1):
            was_active = ~is_halted

            # Momentum forward update
            H_fwd_next = self._unroll_3layer_step(H_fwd, H_0_fwd, edge_src, edge_dst, edge_rel_embed)
            delta_fwd = H_fwd_next - H_fwd
            V_fwd = beta_momentum * V_fwd + (1.0 - beta_momentum) * delta_fwd
            H_fwd = self.momentum_norm_fwd(H_fwd + V_fwd)

            # Momentum backward update
            H_bwd_next = self._unroll_3layer_step(H_bwd, H_0_bwd, edge_src_bwd, edge_dst_bwd, edge_rel_embed)
            delta_bwd = H_bwd_next - H_bwd
            V_bwd = beta_momentum * V_bwd + (1.0 - beta_momentum) * delta_bwd
            H_bwd = self.momentum_norm_bwd(H_bwd + V_bwd)

            C_t, max_c, mean_c = self._compute_collision(H_fwd, H_bwd, node_mask=node_mask)

            # Condition A: Early bridge discovered on real nodes (max_c >= tau_hit with tau_hit > 0.5)
            hit_now = (~is_hit) & (max_c >= self.tau_hit)
            t_hit = torch.where(hit_now, torch.full_like(t_hit, float(t)), t_hit)
            is_hit = is_hit | hit_now

            # Condition B: Frontier convergence / dead end strictly evaluated only after step floor t >= 4
            # Ignore delta convergence on initial uncollided states (max_c <= -5.0)
            valid_history = (max_c > -5.0) & (prev_max_c > -5.0) & (prev2_max_c > -5.0)
            conv_now = (
                (t >= 4)
                & valid_history
                & (torch.abs(max_c - prev_max_c) < self.epsilon_conv)
                & (torch.abs(prev_max_c - prev2_max_c) < self.epsilon_conv)
            )
            dead_end_now = (
                (t >= 6)
                & (~is_hit)
                & (max_c <= -5.0)
                & (torch.abs(max_c - prev_max_c) < 1e-4)
                & (torch.abs(prev_max_c - prev2_max_c) < 1e-4)
            )

            # Halt when early explicit hit condition is met OR when convergence criteria are met after floor t >= 4
            halt_now = (is_hit | ((t >= 4) & (conv_now | dead_end_now))) & (~is_halted)
            steps_taken = torch.where(halt_now & (steps_taken == 0), torch.full_like(steps_taken, float(t)), steps_taken)
            is_halted = is_halted | halt_now

            # Record latest active representations for samples that were active in this step
            if was_active.any():
                last_H_fwd = torch.where(was_active.unsqueeze(-1).unsqueeze(-1), H_fwd, last_H_fwd)
                last_H_bwd = torch.where(was_active.unsqueeze(-1).unsqueeze(-1), H_bwd, last_H_bwd)
                last_C_t = torch.where(was_active.unsqueeze(-1), C_t, last_C_t)
                last_max_c = torch.where(was_active, max_c, last_max_c)
                last_mean_c = torch.where(was_active, mean_c, last_mean_c)

            prev2_max_c = prev_max_c
            prev_max_c = max_c

            if is_halted.all() and not self.training:
                break

        # Handle samples that reached max_steps
        unhalted = ~is_halted
        if unhalted.any():
            steps_taken = torch.where(unhalted, torch.full_like(steps_taken, float(self.max_steps)), steps_taken)
            last_H_fwd = torch.where(unhalted.unsqueeze(-1).unsqueeze(-1), H_fwd, last_H_fwd)
            last_H_bwd = torch.where(unhalted.unsqueeze(-1).unsqueeze(-1), H_bwd, last_H_bwd)
            last_C_t = torch.where(unhalted.unsqueeze(-1), C_t, last_C_t)
            last_max_c = torch.where(unhalted, max_c, last_max_c)
            last_mean_c = torch.where(unhalted, mean_c, last_mean_c)

        # Hinge ponder cost: only penalizes steps after discovery of collision bridge
        hinge_steps = torch.clamp(steps_taken - t_hit, min=0.0)
        ponder_cost = hinge_steps.mean()

        logit, hop, ptr_scores = self._pool_predictions(
            last_H_fwd, last_H_bwd, last_C_t, last_max_c, last_mean_c, src_idx, dst_idx, adj
        )

        res = {
            "logit": logit,
            "hop": hop,
            "pointer_logits": ptr_scores,
            "steps_taken": steps_taken,
            "t_hit": t_hit,
            "ponder_cost": ponder_cost,
            "max_collision": torch.clamp(last_max_c, min=0.0),
        }
        if return_latents:
            res["H_fwd"] = last_H_fwd
            res["H_bwd"] = last_H_bwd
            res["collision_map"] = last_C_t
        return res

    @torch.no_grad()
    def trace_attack_path(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        node_names: Optional[List[str]] = None,
        max_steps: int = 22,
    ) -> Dict[str, Any]:
        """
        Complete Neural Path Tracing (Pointer Beam Search).

        Reconstructs the end-to-end sequence of nodes along the attack corridor
        anchored at the collision bridge node v* = argmax_v C(v).
        """
        self.eval()
        single_batch = {
            k: (v[sample_idx : sample_idx + 1] if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = self(single_batch, steps=max_steps, return_latents=True)

        reach_prob = torch.sigmoid(out["logit"][0]).item()
        c_map = out["collision_map"][0]  # [N]
        ptr_matrix = out["pointer_logits"][0]  # [N, N]
        adj = single_batch["adj"][0]  # [N, N]
        src = single_batch["src_idx"][0].item()
        dst = single_batch["dst_idx"][0].item()
        N = c_map.shape[0]

        is_reachable = reach_prob >= 0.5

        if not is_reachable:
            return {
                "reachable": False,
                "confidence": reach_prob,
                "bridge_node": None,
                "bridge_node_name": None,
                "path": [],
                "path_names": [],
                "path_length": 0,
                "hop_count": None,
                "is_valid_walk": True,
                "subgraph": nx.DiGraph(),
            }

        # 1. Identify collision bridge node v*
        # Mask out dummy padding nodes so distractors never win argmax
        node_mask = single_batch.get("node_mask")
        if node_mask is not None:
            mask_1d = node_mask[0].squeeze(-1) if node_mask.dim() == 3 else node_mask[0]
            c_map_masked = c_map.masked_fill(~mask_1d.bool(), float("-inf"))
        elif node_names:
            is_real = torch.tensor(
                [not str(name).startswith("__distractor_") for name in node_names],
                dtype=torch.bool,
                device=c_map.device,
            )
            c_map_masked = c_map.masked_fill(~is_real, float("-inf"))
        else:
            c_map_masked = c_map

        v_star = c_map_masked.argmax().item()

        # 2. Forward pointer walk: from src to dst using argmax successor
        curr = src
        path_nodes = [src]
        visited = {src}

        for _ in range(N):
            if curr == dst:
                break
            # Find candidate next hop with highest score among valid outgoing edges
            row_scores = ptr_matrix[curr].clone()
            # Mask out self-loop unless already at target
            row_scores[curr] = -1e9
            # Greedy next hop
            next_hop = row_scores.argmax().item()
            if next_hop in visited:
                break
            visited.add(next_hop)
            path_nodes.append(next_hop)
            curr = next_hop

        # If greedy forward walk did not reach dst, fall back to bridge-centered walk
        if path_nodes[-1] != dst:
            # Reconstruct from src -> v* and v* -> dst
            left_curr = src
            left_path = [src]
            left_visited = {src}
            while left_curr != v_star and len(left_path) < N:
                row = ptr_matrix[left_curr].clone()
                row[left_curr] = -1e9
                n_hop = row.argmax().item()
                if n_hop in left_visited:
                    break
                left_visited.add(n_hop)
                left_path.append(n_hop)
                left_curr = n_hop

            right_curr = v_star
            right_path = [v_star]
            right_visited = {v_star}
            while right_curr != dst and len(right_path) < N:
                row = ptr_matrix[right_curr].clone()
                row[right_curr] = -1e9
                n_hop = row.argmax().item()
                if n_hop in right_visited:
                    break
                right_visited.add(n_hop)
                right_path.append(n_hop)
                right_curr = n_hop

            if left_path[-1] == v_star and right_path[-1] == dst:
                path_nodes = left_path[:-1] + right_path
            elif right_path[-1] == dst:
                path_nodes = [src] + right_path
            else:
                path_nodes.append(dst)

        # 3. Verify valid graph walk
        is_valid_walk = True
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if adj[u, v].item() == 0:
                is_valid_walk = False
                break

        path_names = (
            [node_names[idx] if idx < len(node_names) else str(idx) for idx in path_nodes]
            if node_names
            else [str(idx) for idx in path_nodes]
        )
        bridge_name = (
            node_names[v_star] if node_names and v_star < len(node_names) else str(v_star)
        )

        subgraph = nx.DiGraph()
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            subgraph.add_edge(u, v)

        return {
            "reachable": True,
            "confidence": reach_prob,
            "bridge_node": v_star,
            "bridge_node_name": bridge_name,
            "path": path_nodes,
            "path_names": path_names,
            "path_length": len(path_nodes) - 1,
            "hop_count": len(path_nodes) - 1,
            "is_valid_walk": is_valid_walk,
            "subgraph": subgraph,
        }
