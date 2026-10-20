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
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = int(top_k)
        if self.top_k < 1 or self.top_k > num_experts:
            raise ValueError(
                f"top_k must satisfy 1 <= top_k <= num_experts; "
                f"got top_k={self.top_k}, num_experts={num_experts}"
            )
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
        topk_probs, topk_indices = torch.topk(
            probs, self.top_k, dim=-1
        )

        if telemetry_class is not None:
            telemetry_class.record_routing(
                current_branch,
                self.layer_id,
                probs,
                topk_indices,
                valid_tokens,
            )

        if current_branch == "B" and self.training:
            f = torch.bincount(
                topk_indices.reshape(-1), minlength=self.num_experts
            ).float() / (valid_tokens.numel() * self.top_k)
            p_mean = F.softmax(raw_logits.float(), dim=-1).mean(dim=0)
            self.last_aux_loss = self.num_experts * torch.sum(f * p_mean)

        if current_branch == "C" and self.training:
            with torch.no_grad():
                self.accumulated_expert_counts += torch.bincount(
                    topk_indices.reshape(-1), minlength=self.num_experts
                ).float()

        topk_probs = topk_probs / (
            topk_probs.sum(dim=-1, keepdim=True) + 1e-10
        )
        valid_out = torch.zeros_like(valid_x)
        for expert_index, expert in enumerate(self.experts):
            for rank in range(self.top_k):
                selected = topk_indices[:, rank] == expert_index
                if selected.any():
                    expert_out = expert(valid_x[selected])
                    if telemetry_class is not None:
                        telemetry_class.record_expert_output(
                            current_branch,
                            self.layer_id,
                            expert_index,
                            expert_out,
                        )
                    valid_out[selected] += (
                        topk_probs[selected, rank]
                        .to(expert_out.dtype)
                        .unsqueeze(1)
                        * expert_out
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
        top_k: int = 2,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model, n_heads, latent_dim_kv, latent_dim_q, pe_dim
        )
        self.moe = ResearchMoELayer(
            d_model,
            num_experts,
            d_ff,
            vocab_size,
            layer_idx=layer_idx,
            top_k=top_k,
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
        top_k: int = 2,
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
                    top_k=top_k,
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
                "entropy_sum": torch.zeros(
                    1, dtype=torch.float32, device=device
                ),
                # These accumulators are deliberately float64.  They are
                # checkpoint telemetry, not model-state tensors, and the
                # exact mass-conservation invariant matters scientifically.
                "token_affinity": torch.zeros(
                    num_experts,
                    cls._vocab_size,
                    dtype=torch.float64,
                    device=device,
                ),
                "selected_token_counts": torch.zeros(
                    num_experts,
                    cls._vocab_size,
                    dtype=torch.float64,
                    device=device,
                ),
                "top1_counts": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "top2_counts": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "selected_probability_sum": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "selected_probability_count": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "activation_norm_sum": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
                "activation_norm_count": torch.zeros(
                    num_experts, dtype=torch.float64, device=device
                ),
            }

    _vocab_size = 259

    @classmethod
    def configure(cls, vocab_size: int) -> None:
        if int(vocab_size) < 3:
            raise ValueError("Telemetry vocabulary must include PAD/BOS/EOS")
        cls._vocab_size = int(vocab_size)

    @classmethod
    def record_routing(
        cls,
        branch: str,
        layer_id: str,
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        probabilities = probs.detach()
        indices = topk_indices.detach()
        num_tokens, num_experts = probabilities.shape
        if indices.ndim != 2 or indices.shape[0] != num_tokens:
            raise ValueError(
                "Routing index shape is inconsistent with probability shape"
            )
        top_k = int(indices.shape[1])
        if valid_tokens.ndim != 1 or valid_tokens.shape[0] != num_tokens:
            raise ValueError("Token ids are inconsistent with routing rows")
        if valid_tokens.min().item() < 0 or valid_tokens.max().item() >= cls._vocab_size:
            raise ValueError("Token id outside telemetry vocabulary")
        cls._init_layer_buffers(
            branch, layer_id, num_experts, probabilities.device
        )
        accumulator = cls._accumulators[branch][layer_id]
        if "top_k" in accumulator and accumulator["top_k"] != top_k:
            raise ValueError(
                f"top_k changed within telemetry interval: "
                f"{accumulator['top_k']} -> {top_k}"
            )
        accumulator["top_k"] = top_k
        accumulator["tokens_routed"] += num_tokens
        accumulator["hard_assignments"] += torch.bincount(
            indices.reshape(-1), minlength=num_experts
        ).float()
        accumulator["top1_counts"] += torch.bincount(
            indices[:, 0], minlength=num_experts
        ).double()
        accumulator["top2_counts"] += torch.bincount(
            indices.reshape(-1), minlength=num_experts
        ).double()
        normalized_selected = torch.gather(
            probabilities, 1, indices
        )
        normalized_selected = normalized_selected / (
            normalized_selected.sum(dim=-1, keepdim=True) + 1e-12
        )
        # The two weighted CUDA bincount reductions are nondeterministic under
        # the frozen strict reproducibility contract.  Keep the scientific
        # telemetry fields and their precision unchanged, but perform only
        # these two observational reductions on CPU.  The result is copied
        # back to the existing GPU accumulators; no training tensor is
        # modified and no telemetry value enters autograd.
        cpu_indices = indices.reshape(-1).cpu()
        cpu_selected = normalized_selected.reshape(-1).double().cpu()
        cpu_ones = torch.ones(cpu_indices.numel(), dtype=torch.float64)
        selected_probability_sum = torch.bincount(
            cpu_indices,
            weights=cpu_selected,
            minlength=num_experts,
        ).to(device=probabilities.device)
        selected_probability_count = torch.bincount(
            cpu_indices,
            weights=cpu_ones,
            minlength=num_experts,
        ).to(device=probabilities.device)
        accumulator["selected_probability_sum"] += selected_probability_sum
        accumulator["selected_probability_count"] += selected_probability_count
        accumulator["token_affinity"].index_add_(
            1,
            valid_tokens,
            probabilities.double().transpose(0, 1),
        )
        for expert_index in range(num_experts):
            selected_for_expert = (indices == expert_index).sum(dim=1).double()
            if torch.any(selected_for_expert > 0):
                accumulator["selected_token_counts"][expert_index].index_add_(
                    0, valid_tokens, selected_for_expert
                )
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
        accumulator = cls._accumulators[branch][layer_id]
        # Telemetry must never retain the training autograd graph.  This is
        # a read-only measurement path and otherwise leaks a graph per expert
        # dispatch until the next checkpoint.
        norms = expert_output.detach().float().norm(dim=-1).double()
        accumulator["activation_norm_sum"][expert_index] += norms.sum()
        accumulator["activation_norm_count"][expert_index] += norms.numel()

    @classmethod
    def get_summary(cls, branch: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for layer, accumulator in cls._accumulators[branch].items():
            total_tokens = accumulator["tokens_routed"].item()
            if total_tokens == 0:
                continue
            hard_assignments = accumulator["hard_assignments"]
            if hard_assignments.numel() == 1:
                # The existing metric uses sample standard deviation for
                # E>=2.  Sample std is undefined for one expert, so the
                # only valid one-expert CV is exactly zero.
                global_cv_hard = 0.0
            else:
                global_cv_hard = (
                    hard_assignments.std(unbiased=True)
                    / (hard_assignments.mean() + 1e-10)
                ).item()
            max_possible_entropy = math.log(float(hard_assignments.size(0)))
            token_affinity = accumulator["token_affinity"]
            affinity_mass = token_affinity.sum().item()
            if affinity_mass <= 0.0:
                raise ValueError(f"Layer {layer} has zero affinity mass")
            selected_counts = accumulator["selected_token_counts"]
            selected_count_total = selected_counts.sum().item()
            if selected_count_total <= 0.0:
                raise ValueError(f"Layer {layer} has zero selected-token mass")
            selected_probability_count = accumulator[
                "selected_probability_count"
            ]
            selected_probability_sum = accumulator[
                "selected_probability_sum"
            ]
            activation_count = accumulator["activation_norm_count"]
            activation_sum = accumulator["activation_norm_sum"]
            summary[layer] = {
                "top_k": accumulator.get("top_k"),
                "tokens_routed": int(total_tokens),
                "_token_affinity": token_affinity.cpu().tolist(),
                "selected_token_counts": selected_counts.cpu().tolist(),
                "top1_counts": accumulator["top1_counts"].cpu().tolist(),
                "top2_counts": accumulator["top2_counts"].cpu().tolist(),
                "hard_assignments": hard_assignments.cpu().tolist(),
                "selected_probability_sum": selected_probability_sum.cpu().tolist(),
                "selected_probability_count": selected_probability_count.cpu().tolist(),
                "mean_selected_probability": (
                    selected_probability_sum
                    / (selected_probability_count + 1e-12)
                ).cpu().tolist(),
                "activation_norm_sum": activation_sum.cpu().tolist(),
                "activation_norm_count": activation_count.cpu().tolist(),
                "mean_expert_output_norm": (
                    activation_sum / (activation_count + 1e-12)
                ).cpu().tolist(),
                "affinity_mass_total": affinity_mass,
                "selected_token_mass_total": selected_count_total,
                "global_hard_cv": global_cv_hard,
                "max_min_hard_ratio": (
                    hard_assignments.max()
                    / (hard_assignments.min() + 1e-10)
                ).item(),
                "normalized_entropy": (
                    accumulator["entropy_sum"].item() / total_tokens
                )
                / (max_possible_entropy + 1e-10),
            }
        return summary
