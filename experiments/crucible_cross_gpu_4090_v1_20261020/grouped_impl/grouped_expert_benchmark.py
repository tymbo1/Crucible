#!/usr/bin/env python3
"""GPU Neural SSRE v1 deterministic grouped-expert benchmark.

This is a disposable engineering benchmark. Frozen Crucible inputs, manifests
and anchors are read-only. The benchmark performs temporary optimizer updates
in fresh model instances solely to measure complete forward/backward execution
and to test numerical equivalence; no checkpoint is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))

from gpu_step_benchmark import (  # noqa: E402
    assemble_current,
    build_batches,
    load_model,
    loss_sums,
    make_optimizer,
    sha256_file,
    state_digest,
    validate_prefix_inputs,
)


SCHEMA_VERSION = "gpu-neural-ssre-v1-grouped-expert-20261018"
VARIANTS = (
    "original",
    "grouped_rank_preserving",
    "grouped_rank_largest_first",
    "grouped_sorted",
    "batched_padded",
)
GROUPED_VARIANTS = VARIANTS[1:]
FORWARD_TOL = 5e-3
LOSS_TOL = 1e-5
GRAD_TOL = 5e-4
PARAM_TOL = 5e-4


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_file_text(path: Path) -> str:
    return sha256_file(path)


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def capture_state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def compare_state_cpu(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    if set(reference) != set(candidate):
        return {
            "same_keys": False,
            "max_abs": float("inf"),
            "within_tolerance": False,
        }
    max_abs = 0.0
    worst = None
    for name in reference:
        delta = tensor_max_abs(reference[name], candidate[name])
        if delta > max_abs:
            max_abs = delta
            worst = name
    return {
        "same_keys": True,
        "max_abs": max_abs,
        "worst_tensor": worst,
        "tolerance": PARAM_TOL,
        "within_tolerance": max_abs <= PARAM_TOL,
    }


def compare_gradients(
    reference: Mapping[str, torch.Tensor],
    candidate_model: nn.Module,
) -> Dict[str, Any]:
    candidate = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in candidate_model.named_parameters()
        if parameter.grad is not None
    }
    if set(reference) != set(candidate):
        return {
            "same_keys": False,
            "max_abs": float("inf"),
            "within_tolerance": False,
        }
    max_abs = 0.0
    worst = None
    for name in reference:
        delta = tensor_max_abs(reference[name], candidate[name])
        if delta > max_abs:
            max_abs = delta
            worst = name
    return {
        "same_keys": True,
        "max_abs": max_abs,
        "worst_tensor": worst,
        "tolerance": GRAD_TOL,
        "within_tolerance": max_abs <= GRAD_TOL,
    }


def _route_and_aux(
    module: nn.Module,
    valid_x: torch.Tensor,
    valid_tokens: torch.Tensor,
    current_branch: str,
    telemetry_class: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_logits = module.gate(valid_x)
    branch_bias = (
        module.bias_tracking_weights.unsqueeze(0)
        if current_branch == "C"
        else 0.0
    )
    logits = raw_logits + branch_bias
    probs = F.softmax(logits.float(), dim=-1)
    topk_probs, topk_indices = torch.topk(probs, module.top_k, dim=-1)

    if telemetry_class is not None:
        telemetry_class.record_routing(
            current_branch,
            module.layer_id,
            probs,
            topk_indices,
            valid_tokens,
        )

    module.last_aux_loss = torch.tensor(0.0, device=valid_x.device)
    if current_branch == "B" and module.training:
        f = torch.bincount(
            topk_indices.reshape(-1), minlength=module.num_experts
        ).float() / (valid_tokens.numel() * module.top_k)
        p_mean = F.softmax(raw_logits.float(), dim=-1).mean(dim=0)
        module.last_aux_loss = module.num_experts * torch.sum(f * p_mean)

    topk_probs = topk_probs / (
        topk_probs.sum(dim=-1, keepdim=True) + 1e-10
    )
    return topk_probs, topk_indices, probs


def _sorted_assignments(
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    num_experts: int,
) -> Dict[str, Any]:
    token_count, top_k = topk_indices.shape
    flat_experts = topk_indices.reshape(-1)
    flat_weights = topk_probs.reshape(-1)
    flat_ranks = torch.arange(top_k, device=topk_indices.device).view(1, -1)
    flat_ranks = flat_ranks.expand(token_count, -1).reshape(-1)
    flat_tokens = torch.arange(token_count, device=topk_indices.device).view(-1, 1)
    flat_tokens = flat_tokens.expand(-1, top_k).reshape(-1)

    # Sort by expert, then routing rank, then original token position. This
    # preserves the original rank accumulation order for duplicate tokens.
    key = ((flat_experts * top_k + flat_ranks) * token_count) + flat_tokens
    order = torch.argsort(key, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_ranks = flat_ranks.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    counts = torch.bincount(flat_experts, minlength=num_experts)
    pair_counts = torch.bincount(
        flat_experts * top_k + flat_ranks,
        minlength=num_experts * top_k,
    )
    counts_cpu = [int(value) for value in counts.detach().cpu().tolist()]
    pair_counts_cpu = [int(value) for value in pair_counts.detach().cpu().tolist()]
    offsets = [0]
    for count in counts_cpu:
        offsets.append(offsets[-1] + count)
    pair_offsets = [0]
    for count in pair_counts_cpu:
        pair_offsets.append(pair_offsets[-1] + count)
    return {
        "sorted_experts": sorted_experts,
        "sorted_ranks": sorted_ranks,
        "sorted_tokens": sorted_tokens,
        "sorted_weights": sorted_weights,
        "counts": counts,
        "counts_cpu": counts_cpu,
        "pair_counts_cpu": pair_counts_cpu,
        "offsets": offsets,
        "pair_offsets": pair_offsets,
        "token_count": token_count,
        "top_k": top_k,
    }


def grouped_moe_forward(
    module: nn.Module,
    x: torch.Tensor,
    current_tokens: torch.Tensor,
    current_branch: str,
    telemetry_class: Any,
    variant: str,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(-1, hidden_dim)
    tokens_flat = current_tokens.reshape(-1)
    non_pad_mask = tokens_flat != 0
    out_flat = torch.zeros_like(x_flat)
    module.last_aux_loss = torch.tensor(0.0, device=x.device)
    if not non_pad_mask.any():
        return out_flat.view(batch_size, seq_len, hidden_dim)

    valid_x = x_flat[non_pad_mask]
    valid_tokens = tokens_flat[non_pad_mask]
    topk_probs, topk_indices, _ = _route_and_aux(
        module,
        valid_x,
        valid_tokens,
        current_branch,
        telemetry_class,
    )
    assignments = _sorted_assignments(
        topk_indices,
        topk_probs,
        module.num_experts,
    )
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    offsets = assignments["offsets"]
    counts_cpu = assignments["counts_cpu"]
    valid_out = torch.zeros_like(valid_x)

    if variant in ("grouped_rank_largest_first",):
        execution_order = sorted(
            range(module.num_experts),
            key=lambda index: (-counts_cpu[index], index),
        )
    else:
        execution_order = list(range(module.num_experts))

    if variant in ("grouped_rank_preserving", "grouped_rank_largest_first"):
        pair_offsets = assignments["pair_offsets"]
        pair_top_k = assignments["top_k"]
        for expert_index in execution_order:
            for rank in range(pair_top_k):
                pair_id = expert_index * pair_top_k + rank
                start, end = pair_offsets[pair_id], pair_offsets[pair_id + 1]
                if end <= start:
                    continue
                token_indices = sorted_tokens[start:end]
                expert_input = valid_x.index_select(0, token_indices)
                expert_out = module.experts[expert_index](expert_input)
                if telemetry_class is not None:
                    telemetry_class.record_expert_output(
                        current_branch,
                        module.layer_id,
                        expert_index,
                        expert_out,
                    )
                weighted_out = (
                    expert_out
                    * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)
                ).to(valid_out.dtype)
                # Each pair segment has unique token indices. This is the
                # same expert/rank accumulation order as the frozen loop.
                valid_out.index_put_((token_indices,), weighted_out, accumulate=True)
    elif variant == "batched_padded":
        max_count = max(counts_cpu)
        expert_inputs = valid_x.index_select(0, sorted_tokens)
        padded = valid_x.new_zeros(module.num_experts * max_count, hidden_dim)
        positions = torch.arange(sorted_tokens.numel(), device=x.device)
        offset_tensor = torch.tensor(
            offsets[:-1], device=x.device, dtype=positions.dtype
        )
        positions = positions - offset_tensor[assignments["sorted_experts"]]
        destinations = assignments["sorted_experts"] * max_count + positions
        padded.index_copy_(0, destinations, expert_inputs)
        padded = padded.view(module.num_experts, max_count, hidden_dim)

        w1 = torch.stack([expert.w1.weight for expert in module.experts], dim=0)
        w2 = torch.stack([expert.w2.weight for expert in module.experts], dim=0)
        w3 = torch.stack([expert.w3.weight for expert in module.experts], dim=0)
        hidden_1 = torch.bmm(padded, w1.transpose(1, 2))
        hidden_2 = torch.bmm(padded, w2.transpose(1, 2))
        padded_out = torch.bmm(
            F.silu(hidden_1) * hidden_2,
            w3.transpose(1, 2),
        )
        flat_padded_out = padded_out.reshape(-1, hidden_dim)
        sorted_out = flat_padded_out.index_select(0, destinations)
        weighted_out = (
            sorted_out * sorted_weights.to(sorted_out.dtype).unsqueeze(1)
        ).to(valid_out.dtype)
        valid_out.index_add_(0, sorted_tokens, weighted_out)
        if telemetry_class is not None:
            for expert_index in range(module.num_experts):
                start, end = offsets[expert_index], offsets[expert_index + 1]
                if end > start:
                    telemetry_class.record_expert_output(
                        current_branch,
                        module.layer_id,
                        expert_index,
                        sorted_out[start:end],
                    )
    else:
        for expert_index in execution_order:
            start, end = offsets[expert_index], offsets[expert_index + 1]
            if end <= start:
                continue
            token_indices = sorted_tokens[start:end]
            expert_input = valid_x.index_select(0, token_indices)
            expert_out = module.experts[expert_index](expert_input)
            if telemetry_class is not None:
                telemetry_class.record_expert_output(
                    current_branch,
                    module.layer_id,
                    expert_index,
                    expert_out,
                )
            weighted_out = (
                expert_out
                * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)
            ).to(valid_out.dtype)
            valid_out.index_add_(0, token_indices, weighted_out)

    out_flat[non_pad_mask] = valid_out
    return out_flat.view(batch_size, seq_len, hidden_dim)


def patch_grouped_model(model: nn.Module, variant: str) -> None:
    if variant == "original":
        return
    for block in model.layers:
        moe = block.moe

        def wrapped_forward(
            *args: Any,
            _moe=moe,
            _variant=variant,
            **kwargs: Any,
        ):
            return grouped_moe_forward(_moe, *args, variant=_variant, **kwargs)

        moe.forward = wrapped_forward  # type: ignore[method-assign]


def run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    raw: torch.Tensor,
    group_batches: Sequence[Any],
    batch_positions: Mapping[Tuple[int, int], int],
    device: torch.device,
    hp: Mapping[str, Any],
    batches: Sequence[Any],
    timed: bool,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad()
    total_targets = sum(int(item.nonpad_target_tokens) for item in group_batches)
    record = {
        "microbatches": len(group_batches),
        "cpu_batch_prepare_ns": 0,
        "h2d_ns": 0,
        "forward_host_ns": 0,
        "forward_cuda_event_ns": 0,
        "backward_host_ns": 0,
        "backward_cuda_event_ns": 0,
        "optimizer_ns": 0,
        "wall_ns": 0,
        "loss_values": [],
        "valid_targets": 0,
        "raw_targets": 0,
    }
    step_start = now_ns()
    for batch_index, batch in enumerate(group_batches):
        prepare_start = now_ns()
        batch_pos = batch_positions[(int(batch.step), int(batch.microbatch_index))]
        x_cpu, y_cpu = assemble_current(raw, batch)
        record["cpu_batch_prepare_ns"] += now_ns() - prepare_start

        h2d_start = now_ns()
        torch.cuda.synchronize(device)
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        torch.cuda.synchronize(device)
        record["h2d_ns"] += now_ns() - h2d_start

        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device) + (
            batch_index * int(hp["microbatch_size"])
        )

        torch.cuda.synchronize(device)
        forward_host_start = now_ns()
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                x_gpu,
                seq_idx,
                current_branch="B",
                current_sequence_ids=current_sequence_ids,
                telemetry_class=None,
            )
            ce_sum, raw_ce_sum, valid_count, raw_count = loss_sums(logits, y_gpu)
            auxiliary_loss = torch.stack(
                [layer.moe.last_aux_loss for layer in model.layers]
            ).mean()
            loss = (
                ce_sum
                + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count
            ) / total_targets
        forward_end.record()
        forward_end.synchronize()
        record["forward_host_ns"] += now_ns() - forward_host_start
        record["forward_cuda_event_ns"] += int(forward_start.elapsed_time(forward_end) * 1_000_000.0)

        torch.cuda.synchronize(device)
        backward_host_start = now_ns()
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        backward_start.record()
        loss.backward()
        backward_end.record()
        backward_end.synchronize()
        record["backward_host_ns"] += now_ns() - backward_host_start
        record["backward_cuda_event_ns"] += int(backward_start.elapsed_time(backward_end) * 1_000_000.0)
        record["loss_values"].append(float(loss.detach().item()))
        record["valid_targets"] += int(valid_count)
        record["raw_targets"] += int(raw_count)
        del logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids

    clip_start = now_ns()
    nn.utils.clip_grad_norm_(model.parameters(), float(hp["max_norm_clip"]))
    clip_ns = now_ns() - clip_start
    optimizer_start = now_ns()
    torch.cuda.synchronize(device)
    optimizer.step()
    torch.cuda.synchronize(device)
    record["optimizer_ns"] = now_ns() - optimizer_start + clip_ns
    record["wall_ns"] = now_ns() - step_start
    return record


def run_trial(
    *,
    config: Mapping[str, Any],
    root: Path,
    seed: int,
    variant: str,
    device: torch.device,
    raw: torch.Tensor,
    batches: Sequence[Any],
    anchor_path: Path,
    warmup_steps: int,
    timed_steps: int,
) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    model = load_model(anchor_path, hp, device)
    patch_grouped_model(model, variant)
    optimizer = make_optimizer(
        model,
        hp,
        float(config["contract"]["learning_rate_candidates"][0]),
    )
    initial_state = state_digest(model)
    batch_positions = {
        (int(batch.step), int(batch.microbatch_index)): index
        for index, batch in enumerate(batches)
    }
    timed_records: List[Dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for phase, count in (("warmup", warmup_steps), ("timed", timed_steps)):
        for local_step in range(count):
            current_step = local_step + 1
            group_batches = [batch for batch in batches if batch.step == current_step]
            if not group_batches:
                raise ValueError(f"missing schedule step {current_step}")
            record = run_step(
                model,
                optimizer,
                raw,
                group_batches,
                batch_positions,
                device,
                hp,
                batches,
                phase == "timed",
            )
            record["step"] = current_step
            record["phase"] = phase
            if phase == "timed":
                timed_records.append(record)
    torch.cuda.synchronize(device)
    final_state = state_digest(model)
    result = {
        "variant": variant,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "timed_records": timed_records,
        "phase_sums": {
            key: sum(int(record[key]) for record in timed_records)
            for key in (
                "cpu_batch_prepare_ns",
                "h2d_ns",
                "forward_host_ns",
                "forward_cuda_event_ns",
                "backward_host_ns",
                "backward_cuda_event_ns",
                "optimizer_ns",
                "wall_ns",
            )
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "all_finite_losses": all(
            math.isfinite(value)
            for record in timed_records
            for value in record["loss_values"]
        ),
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def equivalence_probe(
    *,
    config: Mapping[str, Any],
    root: Path,
    seed: int,
    variant: str,
    device: torch.device,
    raw: torch.Tensor,
    batches: Sequence[Any],
    anchor_path: Path,
) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    first_step = min(batch.step for batch in batches)
    group_batches = [batch for batch in batches if batch.step == first_step]
    total_targets = sum(int(item.nonpad_target_tokens) for item in group_batches)

    reference_model = load_model(anchor_path, hp, device)
    reference_optimizer = make_optimizer(
        reference_model,
        hp,
        float(config["contract"]["learning_rate_candidates"][0]),
    )
    reference_logits: List[torch.Tensor] = []
    reference_losses: List[float] = []
    reference_model.zero_grad(set_to_none=True)
    for batch in group_batches:
        x_cpu, y_cpu = assemble_current(raw, batch)
        x_gpu, y_gpu = x_cpu.to(device), y_cpu.to(device)
        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = reference_model(
                x_gpu,
                seq_idx,
                current_branch="B",
                current_sequence_ids=current_sequence_ids,
                telemetry_class=None,
            )
            ce_sum, raw_ce_sum, valid_count, raw_count = loss_sums(logits, y_gpu)
            auxiliary_loss = torch.stack(
                [layer.moe.last_aux_loss for layer in reference_model.layers]
            ).mean()
            loss = (
                ce_sum
                + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count
            ) / total_targets
        reference_logits.append(logits.detach().float().cpu())
        reference_losses.append(float(loss.detach().item()))
        loss.backward()
        del logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids
    nn.utils.clip_grad_norm_(reference_model.parameters(), float(hp["max_norm_clip"]))
    reference_gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in reference_model.named_parameters()
        if parameter.grad is not None
    }
    reference_optimizer.step()
    torch.cuda.synchronize(device)
    reference_state = capture_state_cpu(reference_model)
    del reference_optimizer, reference_model
    gc.collect()
    torch.cuda.empty_cache()

    candidate_model = load_model(anchor_path, hp, device)
    patch_grouped_model(candidate_model, variant)
    candidate_optimizer = make_optimizer(
        candidate_model,
        hp,
        float(config["contract"]["learning_rate_candidates"][0]),
    )
    candidate_losses: List[float] = []
    max_logits_abs = 0.0
    candidate_model.zero_grad(set_to_none=True)
    for index, batch in enumerate(group_batches):
        x_cpu, y_cpu = assemble_current(raw, batch)
        x_gpu, y_gpu = x_cpu.to(device), y_cpu.to(device)
        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = candidate_model(
                x_gpu,
                seq_idx,
                current_branch="B",
                current_sequence_ids=current_sequence_ids,
                telemetry_class=None,
            )
            ce_sum, raw_ce_sum, valid_count, raw_count = loss_sums(logits, y_gpu)
            auxiliary_loss = torch.stack(
                [layer.moe.last_aux_loss for layer in candidate_model.layers]
            ).mean()
            loss = (
                ce_sum
                + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count
            ) / total_targets
        max_logits_abs = max(
            max_logits_abs,
            tensor_max_abs(logits.detach().float().cpu(), reference_logits[index]),
        )
        candidate_losses.append(float(loss.detach().item()))
        loss.backward()
        del logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids
    nn.utils.clip_grad_norm_(candidate_model.parameters(), float(hp["max_norm_clip"]))
    gradient_report = compare_gradients(reference_gradients, candidate_model)
    candidate_optimizer.step()
    torch.cuda.synchronize(device)
    candidate_state = capture_state_cpu(candidate_model)
    state_report = compare_state_cpu(reference_state, candidate_state)
    loss_delta = max(
        (abs(a - b) for a, b in zip(reference_losses, candidate_losses)),
        default=0.0,
    )
    report = {
        "variant": variant,
        "forward": {
            "max_logits_abs": max_logits_abs,
            "tolerance": FORWARD_TOL,
            "within_tolerance": max_logits_abs <= FORWARD_TOL,
        },
        "loss": {
            "max_abs": loss_delta,
            "tolerance": LOSS_TOL,
            "within_tolerance": loss_delta <= LOSS_TOL,
        },
        "gradients": gradient_report,
        "post_optimizer_parameters": state_report,
        "accepted": (
            max_logits_abs <= FORWARD_TOL
            and loss_delta <= LOSS_TOL
            and gradient_report["within_tolerance"]
            and state_report["within_tolerance"]
        ),
    }
    del candidate_optimizer, candidate_model, reference_logits, reference_losses
    del reference_gradients, reference_state, candidate_state
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic grouped-expert E16 benchmark")
    parser.add_argument(
        "--context-root",
        type=Path,
        default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required by the frozen E16 contract")
    # Match the validated v4 deterministic control. CUDA bincount is an
    # aggregate routing operation without a deterministic PyTorch kernel, so
    # warn_only is intentional; the model path and SDPA implementation remain
    # deterministic for the equivalence gate.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda")
    root = args.context_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    if (
        int(hp["num_experts"]) != 16
        or int(hp["top_k"]) != 2
        or config["contract"]["architectural_branches"] != ["B"]
    ):
        raise ValueError("frozen E16 Branch-B geometry mismatch")
    anchor_path = root / "crucible_manifest" / config["anchors"][str(args.seed)]["relative_path"]
    anchor_sha = sha256_file(anchor_path)
    if anchor_sha != config["anchors"][str(args.seed)]["binary_payload_sha256"]:
        raise ValueError("anchor SHA mismatch")
    max_steps = args.warmup_steps + args.timed_steps
    batches, schedule_report, raw, source_report, binary_report = build_batches(
        config, root, args.seed, max_steps
    )
    index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_report = validate_prefix_inputs(raw, batches, index_tensors, min(32, len(batches)))

    equivalence = {}
    for variant in GROUPED_VARIANTS:
        equivalence[variant] = equivalence_probe(
            config=config,
            root=root,
            seed=args.seed,
            variant=variant,
            device=device,
            raw=raw,
            batches=batches,
            anchor_path=anchor_path,
        )
        print(json.dumps({"equivalence": variant, "accepted": equivalence[variant]["accepted"]}, sort_keys=True), flush=True)

    runs: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for variant in VARIANTS:
            result = run_trial(
                config=config,
                root=root,
                seed=args.seed,
                variant=variant,
                device=device,
                raw=raw,
                batches=batches,
                anchor_path=anchor_path,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
            )
            result["repetition"] = repetition
            runs.append(result)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "repetition": repetition,
                        "timed_wall_s": round(result["phase_sums"]["wall_ns"] / 1e9, 4),
                        "forward_cuda_s": round(result["phase_sums"]["forward_cuda_event_ns"] / 1e9, 4),
                        "backward_cuda_s": round(result["phase_sums"]["backward_cuda_event_ns"] / 1e9, 4),
                        "peak_vram_gib": round(result["peak_cuda_allocated_bytes"] / (1024**3), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "context_root": str(root),
        "config_sha256": sha256_file(config_path),
        "anchor_sha256": anchor_sha,
        "source_report": source_report,
        "binary_report": binary_report,
        "schedule_report": schedule_report,
        "input_report": input_report,
        "parameters": {
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "repetitions": args.repetitions,
            "model_parameter_count": int(hp["expected_parameter_geometry"]),
            "context": int(hp["context_window_length"]),
            "num_experts": int(hp["num_experts"]),
            "top_k": int(hp["top_k"]),
            "equivalence_tolerances": {
                "forward_abs": FORWARD_TOL,
                "loss_abs": LOSS_TOL,
                "gradient_abs": GRAD_TOL,
                "parameter_abs": PARAM_TOL,
            },
        },
        "equivalence": equivalence,
        "runs": runs,
        "integrity": {
            "input_prefix_exact": input_report["exact_equivalence"],
            "all_grouped_equivalence_pass": all(item["accepted"] for item in equivalence.values()),
            "all_runs_finite": all(item["all_finite_losses"] for item in runs),
            "optimizer_step_is_disposable": True,
        },
        "limitations": [
            "This is a local engineering benchmark, not a scientific training run.",
            "The grouped variants retain a Python loop over expert modules; they are deterministic baselines, not custom fused CUDA kernels.",
            "The batched padded arm intentionally measures padding waste and is not presumed to win.",
            "No neural scheduling controller was trained or evaluated.",
        ],
    }
    out_path = output / "grouped_expert_benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "grouped_expert_benchmark_summary.json": sha256_file(out_path),
        "frozen_config_sha256": summary["config_sha256"],
        "frozen_anchor_sha256": anchor_sha,
        "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    (output / "benchmark_output_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_path), "all_equivalence_pass": summary["integrity"]["all_grouped_equivalence_pass"], "runs": len(runs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

