import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_dim_kv: int,
        latent_dim_q: int,
        pe_dim: int,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if pe_dim % 2 != 0:
            raise ValueError("pe_dim must be even")
        self.n_heads = n_heads
        self.pe_dim = pe_dim
        self.head_dim = d_model // n_heads
        self.q_down_proj = nn.Linear(d_model, latent_dim_q, bias=False)
        self.q_up_proj = nn.Linear(latent_dim_q, n_heads * self.head_dim, bias=False)
        self.q_pe_proj = nn.Linear(latent_dim_q, n_heads * pe_dim, bias=False)
        self.kv_down_proj = nn.Linear(d_model, latent_dim_kv, bias=False)
        self.k_up_proj = nn.Linear(latent_dim_kv, n_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(latent_dim_kv, n_heads * self.head_dim, bias=False)
        self.k_pe_proj = nn.Linear(d_model, pe_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_idx: torch.Tensor,
        current_sequence_ids: Optional[torch.Tensor] = None,
        cache: Any = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q_latent = self.q_down_proj(x)
        q_heads = self.q_up_proj(q_latent).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.pe_dim, 2, device=x.device).float()
                / self.pe_dim
            )
        )
        sinusoid_inp = torch.einsum("b s, f -> b s f", seq_idx.float(), inv_freq)
        sin = torch.sin(sinusoid_inp).repeat(1, 1, 2).unsqueeze(2).to(x.dtype)
        cos = torch.cos(sinusoid_inp).repeat(1, 1, 2).unsqueeze(2).to(x.dtype)

        q_pe_raw = self.q_pe_proj(q_latent).view(
            batch_size, seq_len, self.n_heads, self.pe_dim
        )
        q_pe_rot = torch.cat(
            [-q_pe_raw[..., self.pe_dim // 2 :], q_pe_raw[..., : self.pe_dim // 2]],
            dim=-1,
        )
        q_pe = q_pe_raw * cos + q_pe_rot * sin

        compressed_kv = self.kv_down_proj(x)
        k_pe_raw = self.k_pe_proj(x).unsqueeze(2)
        k_pe_rot = torch.cat(
            [-k_pe_raw[..., self.pe_dim // 2 :], k_pe_raw[..., : self.pe_dim // 2]],
            dim=-1,
        )
        k_pe = (k_pe_raw * cos + k_pe_rot * sin).squeeze(2)

        if cache is not None:
            (
                full_compressed_kv,
                full_k_pe,
                cached_positions,
                past_len,
            ) = cache.update(
                compressed_kv, k_pe, seq_idx, current_sequence_ids
            )
        else:
            full_compressed_kv = compressed_kv
            full_k_pe = k_pe
            cached_positions = seq_idx
            past_len = 0

        kv_seq_len = full_compressed_kv.size(1)
        k_heads = self.k_up_proj(full_compressed_kv).view(
            batch_size, kv_seq_len, self.n_heads, self.head_dim
        )
        v_heads = self.v_up_proj(full_compressed_kv).view(
            batch_size, kv_seq_len, self.n_heads, self.head_dim
        )

        q = torch.cat([q_heads, q_pe], dim=-1).transpose(1, 2)
        k = torch.cat(
            [
                k_heads,
                full_k_pe.unsqueeze(2).expand(
                    -1, -1, self.n_heads, -1
                ),
            ],
            dim=-1,
        ).transpose(1, 2)
        v = v_heads.transpose(1, 2)

        q_idx = torch.arange(
            past_len, past_len + seq_len, device=x.device
        ).unsqueeze(-1)
        k_idx = torch.arange(kv_seq_len, device=x.device).unsqueeze(0)
        combined_mask = q_idx >= k_idx
        if cache is not None and cache.max_cache_len is not None:
            combined_mask = combined_mask & (
                k_idx >= (q_idx - cache.max_cache_len + 1)
            )

        final_mask = combined_mask.unsqueeze(0) & (
            seq_idx.unsqueeze(-1) >= cached_positions.unsqueeze(1)
        )
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=final_mask.unsqueeze(1)
        )

        if cache is not None:
            cache.evict_history_()

        return self.out_proj(
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )


class ResearchMoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_ff: int,
        vocab_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.layer_id = f"layer_{layer_idx}_moe"
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff) for _ in range(num_experts)]
        )
        self.register_buffer("bias_tracking_weights", torch.zeros(num_experts))
        self.register_buffer("accumulated_expert_counts", torch.zeros(num_experts))
        self.last_aux_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        current_tokens: torch.Tensor,
        current_branch: str,
        telemetry_class: Any,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        tokens_flat = current_tokens.reshape(-1)
        non_pad_mask = tokens_flat != 0
        out_flat = torch.zeros_like(x_flat)

        self.last_aux_loss = torch.tensor(0.0, device=x.device)
        if not non_pad_mask.any():
            return out_flat.view(batch_size, seq_len, hidden_dim)

        valid_x = x_flat[non_pad_mask]
        valid_tokens = tokens_flat[non_pad_mask]
        raw_logits = self.gate(valid_x)
        branch_bias = (
            self.bias_tracking_weights.unsqueeze(0)
            if current_branch == "C"
            else 0.0
        )
        logits = raw_logits + branch_bias
        probs = F.softmax(logits.float(), dim=-1)
        top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)

        if telemetry_class is not None:
            telemetry_class.record_routing(
                current_branch,
                self.layer_id,
                probs,
                top2_indices,
                valid_tokens,
                self.vocab_size,
            )

        if current_branch == "B" and self.training:
            f = torch.bincount(
                top2_indices.reshape(-1), minlength=self.num_experts
            ).float() / (valid_tokens.numel() * 2)
            p_mean = F.softmax(raw_logits.float(), dim=-1).mean(dim=0)
            self.last_aux_loss = self.num_experts * torch.sum(f * p_mean)

        if current_branch == "C" and self.training:
            with torch.no_grad():
                self.accumulated_expert_counts += torch.bincount(
                    top2_indices.reshape(-1), minlength=self.num_experts
                ).float()

        top2_probs = top2_probs / (
            top2_probs.sum(dim=-1, keepdim=True) + 1e-10
        )
        valid_out = torch.zeros_like(valid_x)
        for expert_index, expert in enumerate(self.experts):
            mask1 = top2_indices[:, 0] == expert_index
            mask2 = top2_indices[:, 1] == expert_index
            if mask1.any():
                expert_out1 = expert(valid_x[mask1])
                if telemetry_class is not None:
                    telemetry_class.record_expert_output(
                        current_branch,
                        self.layer_id,
                        expert_index,
                        expert_out1,
                    )
                valid_out[mask1] += (
                    top2_probs[mask1, 0]
                    .to(expert_out1.dtype)
                    .unsqueeze(1)
                    * expert_out1
                )
            if mask2.any():
                expert_out2 = expert(valid_x[mask2])
                if telemetry_class is not None:
                    telemetry_class.record_expert_output(
                        current_branch,
                        self.layer_id,
                        expert_index,
                        expert_out2,
                    )
                valid_out[mask2] += (
                    top2_probs[mask2, 1]
                    .to(expert_out2.dtype)
                    .unsqueeze(1)
                    * expert_out2
                )

        out_flat[non_pad_mask] = valid_out
        return out_flat.view(batch_size, seq_len, hidden_dim)

    def execute_global_bias_step_update_(
        self, step_size: float = 0.01
    ) -> Tuple[float, List[float]]:
        with torch.no_grad():
            total_tokens = self.accumulated_expert_counts.sum()
            error_list = [0.0] * self.num_experts
            bias_delta = 0.0
            if total_tokens > 0:
                actual_frac = self.accumulated_expert_counts / (total_tokens + 1e-10)
                error = (1.0 / self.num_experts) - actual_frac
                error_list = error.cpu().tolist()
                delta_tensor = step_size * error
                bias_delta = delta_tensor.abs().mean().item()
                self.bias_tracking_weights += delta_tensor
            self.accumulated_expert_counts.zero_()
        return bias_delta, error_list


class ExperimentalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_dim_kv: int,
        latent_dim_q: int,
        pe_dim: int,
        num_experts: int,
        d_ff: int,
        vocab_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model, n_heads, latent_dim_kv, latent_dim_q, pe_dim
        )
        self.moe = ResearchMoELayer(
            d_model, num_experts, d_ff, vocab_size, layer_idx=layer_idx
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_idx: torch.Tensor,
        current_tokens: torch.Tensor,
        current_branch: str,
        current_sequence_ids: Optional[torch.Tensor] = None,
        cache: Any = None,
        telemetry_class: Any = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            seq_idx,
            current_sequence_ids=current_sequence_ids,
            cache=cache,
        )
        return x + self.moe(
            self.ffn_norm(x),
            current_tokens=current_tokens,
            current_branch=current_branch,
            telemetry_class=telemetry_class,
        )


class BraidPrototypeLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        latent_dim_kv: int,
        latent_dim_q: int,
        pe_dim: int,
        num_experts: int,
        d_ff: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ExperimentalTransformerBlock(
                    d_model,
                    n_heads,
                    latent_dim_kv,
                    latent_dim_q,
                    pe_dim,
                    num_experts,
                    d_ff,
                    vocab_size,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_idx: torch.Tensor,
        current_branch: str = "A",
        current_sequence_ids: Optional[torch.Tensor] = None,
        caches: Optional[List[Any]] = None,
        telemetry_class: Any = None,
    ) -> torch.Tensor:
        x = self.token_emb(tokens)
        for index, layer in enumerate(self.layers):
            layer_cache = caches[index] if caches is not None else None
            x = layer(
                x,
                seq_idx,
                current_tokens=tokens,
                current_branch=current_branch,
                current_sequence_ids=current_sequence_ids,
                cache=layer_cache,
                telemetry_class=telemetry_class,
            )
        return self.lm_head(self.final_norm(x))


class ResearchMoETelemetry:
    _accumulators: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
        "A": {},
        "B": {},
        "C": {},
    }

    @classmethod
    def reset_step_buffer(cls) -> None:
        cls._accumulators = {"A": {}, "B": {}, "C": {}}

    @classmethod
    def _init_layer_buffers(
        cls,
        branch: str,
        layer_id: str,
        num_experts: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        if layer_id not in cls._accumulators[branch]:
            cls._accumulators[branch][layer_id] = {
                "tokens_routed": torch.zeros(
                    1, dtype=torch.long, device=device
                ),
                "hard_assignments": torch.zeros(
                    num_experts, dtype=torch.float32, device=device
                ),
                "top1_assignments": torch.zeros(
                    num_experts, dtype=torch.long, device=device
                ),
                "top2_assignments": torch.zeros(
                    num_experts, dtype=torch.long, device=device
                ),
                "token_affinity": torch.zeros(
                    (num_experts, vocab_size),
                    dtype=torch.float64,
                    device=device,
                ),
                "selected_probability_sum": torch.zeros(
                    num_experts, dtype=torch.float32, device=device
                ),
                "selected_probability_count": torch.zeros(
                    num_experts, dtype=torch.long, device=device
                ),
                "expert_output_norm_sum": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "expert_output_norm_count": torch.zeros(
                    num_experts, dtype=torch.long, device=device
                ),
                "population_overlap_sum": torch.zeros(
                    (num_experts, num_experts),
                    dtype=torch.float32,
                    device=device,
                ),
                "population_overlap_batches": torch.zeros(
                    1, dtype=torch.long, device=device
                ),
                "entropy_sum": torch.zeros(
                    1, dtype=torch.float32, device=device
                ),
            }

    @classmethod
    def record_routing(
        cls,
        branch: str,
        layer_id: str,
        probs: torch.Tensor,
        top2_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
        vocab_size: int,
    ) -> None:
        probabilities = probs.detach()
        indices = top2_indices.detach()
        tokens = valid_tokens.detach()
        num_tokens, num_experts = probabilities.shape
        cls._init_layer_buffers(
            branch,
            layer_id,
            num_experts,
            vocab_size,
            probabilities.device,
        )
        accumulator = cls._accumulators[branch][layer_id]
        if tokens.numel() and (
            int(tokens.min().item()) < 0
            or int(tokens.max().item()) >= vocab_size
        ):
            raise RuntimeError(
                f"{layer_id}: token id outside telemetry vocabulary"
            )
        accumulator["tokens_routed"] += num_tokens
        top1_counts = torch.bincount(
            indices[:, 0], minlength=num_experts
        )
        top2_counts = torch.bincount(
            indices.reshape(-1), minlength=num_experts
        )
        accumulator["hard_assignments"] += top2_counts.float()
        accumulator["top1_assignments"] += top1_counts
        accumulator["top2_assignments"] += top2_counts
        accumulator["token_affinity"].scatter_add_(
            1,
            tokens.unsqueeze(0).expand(num_experts, -1),
            probabilities.transpose(0, 1).double(),
        )
        selected_probabilities = probabilities.gather(1, indices)
        accumulator["selected_probability_sum"] += torch.zeros(
            num_experts, dtype=torch.float32, device=probabilities.device
        ).scatter_add_(
            0,
            indices.reshape(-1),
            selected_probabilities.reshape(-1).float(),
        )
        accumulator["selected_probability_count"] += top2_counts
        for first_expert in range(num_experts):
            first_mask = (indices == first_expert).any(dim=1)
            if not first_mask.any():
                continue
            for second_expert in range(first_expert + 1, num_experts):
                second_mask = (indices == second_expert).any(dim=1)
                union = first_mask | second_mask
                if union.any():
                    intersection = (first_mask & second_mask).sum().float()
                    accumulator["population_overlap_sum"][
                        first_expert, second_expert
                    ] += intersection / union.sum().float()
                    accumulator["population_overlap_sum"][
                        second_expert, first_expert
                    ] += intersection / union.sum().float()
        accumulator["population_overlap_batches"] += 1
        accumulator["entropy_sum"] += -torch.sum(
            probabilities * torch.log(probabilities + 1e-10), dim=-1
        ).sum()

    @classmethod
    def record_expert_output(
        cls,
        branch: str,
        layer_id: str,
        expert_index: int,
        expert_output: torch.Tensor,
    ) -> None:
        output = expert_output.detach().float()
        accumulator = cls._accumulators[branch][layer_id]
        accumulator["expert_output_norm_sum"][expert_index] += torch.linalg.vector_norm(
            output, dim=-1
        ).double().sum()
        accumulator["expert_output_norm_count"][expert_index] += output.size(0)

    @classmethod
    def get_summary(cls, branch: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for layer, accumulator in cls._accumulators[branch].items():
            total_tokens = accumulator["tokens_routed"].item()
            if total_tokens == 0:
                continue
            hard_assignments = accumulator["hard_assignments"]
            top1_assignments = accumulator["top1_assignments"]
            top2_assignments = accumulator["top2_assignments"]
            if int(top1_assignments.sum().item()) != int(total_tokens):
                raise RuntimeError(f"{layer}: top-1 assignment count mismatch")
            if int(top2_assignments.sum().item()) != int(2 * total_tokens):
                raise RuntimeError(f"{layer}: top-2 assignment count mismatch")
            global_cv_hard = (
                hard_assignments.std(unbiased=True)
                / (hard_assignments.mean() + 1e-10)
            ).item()
            max_possible_entropy = math.log(float(hard_assignments.size(0)))
            top1_share = top1_assignments.float() / max(1, total_tokens)
            top2_share = top2_assignments.float() / max(1, 2 * total_tokens)
            selected_probability_mean = accumulator[
                "selected_probability_sum"
            ] / accumulator["selected_probability_count"].clamp_min(1).float()
            expert_output_norm_mean = accumulator[
                "expert_output_norm_sum"
            ] / accumulator["expert_output_norm_count"].clamp_min(1).double()
            overlap_batches = accumulator["population_overlap_batches"].clamp_min(1).float()
            summary[layer] = {
                "global_hard_cv": global_cv_hard,
                "max_min_hard_ratio": (
                    hard_assignments.max()
                    / (hard_assignments.min() + 1e-10)
                ).item(),
                "normalized_entropy": (
                    accumulator["entropy_sum"].item() / total_tokens
                )
                / (max_possible_entropy + 1e-10),
                "specialization": {
                    "top1_token_count": top1_assignments.cpu().tolist(),
                    "top2_token_count": top2_assignments.cpu().tolist(),
                    "top1_token_share": top1_share.cpu().tolist(),
                    "top2_token_share": top2_share.cpu().tolist(),
                    "top1_share_sum": top1_share.sum().item(),
                    "top2_share_sum": top2_share.sum().item(),
                    "_token_affinity": accumulator[
                        "token_affinity"
                    ].cpu().tolist(),
                    "selected_mean_router_probability": selected_probability_mean.cpu().tolist(),
                    "expert_output_l2_mean": expert_output_norm_mean.cpu().tolist(),
                    "selected_token_population_overlap": (
                        accumulator["population_overlap_sum"] / overlap_batches
                    ).cpu().tolist(),
                },
            }
        return summary
