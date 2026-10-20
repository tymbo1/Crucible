#!/usr/bin/env python3
"""GPU Neural SSRE v2 full-model strategy landscape benchmark.

This benchmark imports the accepted v1 execution implementation read-only,
adds a rank-major grouped candidate, and extends the equivalence gate to hash
optimizer state. It never writes to a frozen Crucible package.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))

import grouped_expert_benchmark as v1  # noqa: E402


SCHEMA_VERSION = "gpu-neural-ssre-v2-strategy-landscape-20261018"
CONTEXT_NAMES = ("context_128", "context_256", "context_512")
SEEDS = (42, 1337)
FULL_VARIANTS = (
    "original",
    "grouped_rank_preserving",
    "grouped_rank_largest_first",
    "grouped_rank_major",
)
STRICT_CANDIDATES = FULL_VARIANTS[1:]
FORWARD_TOL = v1.FORWARD_TOL
LOSS_TOL = v1.LOSS_TOL
GRAD_TOL = v1.GRAD_TOL
PARAM_TOL = v1.PARAM_TOL


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def optimizer_state_digest(optimizer: torch.optim.Optimizer) -> str:
    """Hash optimizer configuration and every serialized optimizer tensor."""
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    groups = []
    for group in state["param_groups"]:
        groups.append({key: value for key, value in group.items() if key != "params"})
        groups[-1]["params"] = list(group["params"])
    digest.update(json.dumps(groups, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for param_id in sorted(state["state"], key=int):
        digest.update(f"param:{int(param_id)}\n".encode("ascii"))
        slot = state["state"][param_id]
        for name in sorted(slot):
            value = slot[name]
            digest.update(f"slot:{name}\n".encode("utf-8"))
            if isinstance(value, torch.Tensor):
                value_cpu = value.detach().cpu().contiguous()
                digest.update(str(value_cpu.dtype).encode("ascii"))
                digest.update(str(tuple(value_cpu.shape)).encode("ascii"))
                digest.update(value_cpu.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def capture_state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def compare_state_cpu(reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    if set(reference) != set(candidate):
        return {"same_keys": False, "max_abs": float("inf"), "within_tolerance": False}
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


def compare_gradients(reference: Mapping[str, torch.Tensor], candidate_model: nn.Module) -> Dict[str, Any]:
    candidate = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in candidate_model.named_parameters()
        if parameter.grad is not None
    }
    if set(reference) != set(candidate):
        return {"same_keys": False, "max_abs": float("inf"), "within_tolerance": False}
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


def rank_major_assignments(topk_indices: torch.Tensor, topk_probs: torch.Tensor, num_experts: int) -> Dict[str, Any]:
    token_count, top_k = topk_indices.shape
    flat_experts = topk_indices.reshape(-1)
    flat_weights = topk_probs.reshape(-1)
    flat_ranks = torch.arange(top_k, device=topk_indices.device).view(1, -1)
    flat_ranks = flat_ranks.expand(token_count, -1).reshape(-1)
    flat_tokens = torch.arange(token_count, device=topk_indices.device).view(-1, 1)
    flat_tokens = flat_tokens.expand(-1, top_k).reshape(-1)
    key = ((flat_ranks * num_experts + flat_experts) * token_count) + flat_tokens
    order = torch.argsort(key, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_ranks = flat_ranks.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    pair_counts = torch.bincount(
        sorted_ranks * num_experts + sorted_experts,
        minlength=top_k * num_experts,
    )
    pair_counts_cpu = [int(value) for value in pair_counts.detach().cpu().tolist()]
    pair_offsets = [0]
    for count in pair_counts_cpu:
        pair_offsets.append(pair_offsets[-1] + count)
    return {
        "sorted_experts": sorted_experts,
        "sorted_ranks": sorted_ranks,
        "sorted_tokens": sorted_tokens,
        "sorted_weights": sorted_weights,
        "pair_counts_cpu": pair_counts_cpu,
        "pair_offsets": pair_offsets,
        "token_count": token_count,
        "top_k": top_k,
    }


def grouped_rank_major_moe_forward(
    module: nn.Module,
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
    module.last_aux_loss = torch.tensor(0.0, device=x.device)
    if not non_pad_mask.any():
        return out_flat.view(batch_size, seq_len, hidden_dim)
    valid_x = x_flat[non_pad_mask]
    valid_tokens = tokens_flat[non_pad_mask]
    topk_probs, topk_indices, _ = v1._route_and_aux(
        module, valid_x, valid_tokens, current_branch, telemetry_class
    )
    assignments = rank_major_assignments(topk_indices, topk_probs, module.num_experts)
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    pair_offsets = assignments["pair_offsets"]
    top_k = assignments["top_k"]
    valid_out = torch.zeros_like(valid_x)
    for rank in range(top_k):
        for expert_index in range(module.num_experts):
            pair_id = rank * module.num_experts + expert_index
            start, end = pair_offsets[pair_id], pair_offsets[pair_id + 1]
            if end <= start:
                continue
            token_indices = sorted_tokens[start:end]
            expert_out = module.experts[expert_index](valid_x.index_select(0, token_indices))
            if telemetry_class is not None:
                telemetry_class.record_expert_output(
                    current_branch, module.layer_id, expert_index, expert_out
                )
            weighted_out = (
                expert_out
                * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)
            ).to(valid_out.dtype)
            valid_out.index_put_((token_indices,), weighted_out, accumulate=True)
    out_flat[non_pad_mask] = valid_out
    return out_flat.view(batch_size, seq_len, hidden_dim)


def patch_model(model: nn.Module, variant: str) -> None:
    if variant != "grouped_rank_major":
        v1.patch_grouped_model(model, variant)
        return
    for block in model.layers:
        moe = block.moe

        def wrapped_forward(*args: Any, _moe=moe, **kwargs: Any):
            return grouped_rank_major_moe_forward(_moe, *args, **kwargs)

        moe.forward = wrapped_forward  # type: ignore[method-assign]


def run_disposable_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    raw: torch.Tensor,
    group_batches: Sequence[Any],
    batch_positions: Mapping[Tuple[int, int], int],
    device: torch.device,
    hp: Mapping[str, Any],
) -> Dict[str, Any]:
    return v1.run_step(
        model,
        optimizer,
        raw,
        group_batches,
        batch_positions,
        device,
        hp,
        group_batches,
        True,
    )


def equivalence_probe(
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

    reference_model = v1.load_model(anchor_path, hp, device)
    reference_optimizer = v1.make_optimizer(
        reference_model, hp, float(config["contract"]["learning_rate_candidates"][0])
    )
    reference_logits: List[torch.Tensor] = []
    reference_losses: List[float] = []
    reference_model.zero_grad(set_to_none=True)
    for batch in group_batches:
        x_cpu, y_cpu = v1.assemble_current(raw, batch)
        x_gpu, y_gpu = x_cpu.to(device), y_cpu.to(device)
        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = reference_model(
                x_gpu, seq_idx, current_branch="B", current_sequence_ids=current_sequence_ids, telemetry_class=None
            )
            ce_sum, _, valid_count, _ = v1.loss_sums(logits, y_gpu)
            auxiliary_loss = torch.stack([layer.moe.last_aux_loss for layer in reference_model.layers]).mean()
            loss = (ce_sum + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count) / total_targets
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
    reference_model_hash = v1.state_digest(reference_model)
    reference_optimizer_hash = optimizer_state_digest(reference_optimizer)
    del reference_optimizer, reference_model
    gc.collect()
    torch.cuda.empty_cache()

    candidate_model = v1.load_model(anchor_path, hp, device)
    patch_model(candidate_model, variant)
    candidate_optimizer = v1.make_optimizer(
        candidate_model, hp, float(config["contract"]["learning_rate_candidates"][0])
    )
    candidate_losses: List[float] = []
    max_logits_abs = 0.0
    candidate_model.zero_grad(set_to_none=True)
    for index, batch in enumerate(group_batches):
        x_cpu, y_cpu = v1.assemble_current(raw, batch)
        x_gpu, y_gpu = x_cpu.to(device), y_cpu.to(device)
        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = candidate_model(
                x_gpu, seq_idx, current_branch="B", current_sequence_ids=current_sequence_ids, telemetry_class=None
            )
            ce_sum, _, valid_count, _ = v1.loss_sums(logits, y_gpu)
            auxiliary_loss = torch.stack([layer.moe.last_aux_loss for layer in candidate_model.layers]).mean()
            loss = (ce_sum + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count) / total_targets
        max_logits_abs = max(max_logits_abs, tensor_max_abs(logits.detach().float().cpu(), reference_logits[index]))
        candidate_losses.append(float(loss.detach().item()))
        loss.backward()
        del logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids
    nn.utils.clip_grad_norm_(candidate_model.parameters(), float(hp["max_norm_clip"]))
    gradient_report = compare_gradients(reference_gradients, candidate_model)
    candidate_optimizer.step()
    torch.cuda.synchronize(device)
    candidate_state = capture_state_cpu(candidate_model)
    state_report = compare_state_cpu(reference_state, candidate_state)
    candidate_model_hash = v1.state_digest(candidate_model)
    candidate_optimizer_hash = optimizer_state_digest(candidate_optimizer)
    loss_delta = max((abs(a - b) for a, b in zip(reference_losses, candidate_losses)), default=0.0)
    optimizer_equal = reference_optimizer_hash == candidate_optimizer_hash
    report = {
        "variant": variant,
        "forward": {"max_logits_abs": max_logits_abs, "tolerance": FORWARD_TOL, "within_tolerance": max_logits_abs <= FORWARD_TOL},
        "loss": {"max_abs": loss_delta, "tolerance": LOSS_TOL, "within_tolerance": loss_delta <= LOSS_TOL},
        "gradients": gradient_report,
        "post_optimizer_parameters": state_report,
        "model_state_sha256": {"reference": reference_model_hash, "candidate": candidate_model_hash, "equal": reference_model_hash == candidate_model_hash},
        "optimizer_state_sha256": {"reference": reference_optimizer_hash, "candidate": candidate_optimizer_hash, "equal": optimizer_equal},
        "accepted": (
            max_logits_abs <= FORWARD_TOL
            and loss_delta <= LOSS_TOL
            and gradient_report["within_tolerance"]
            and state_report["within_tolerance"]
            and optimizer_equal
        ),
    }
    del candidate_optimizer, candidate_model, reference_logits, reference_losses
    del reference_gradients, reference_state, candidate_state
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_trial(
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
    repetition: int,
) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    model = v1.load_model(anchor_path, hp, device)
    patch_model(model, variant)
    optimizer = v1.make_optimizer(model, hp, float(config["contract"]["learning_rate_candidates"][0]))
    initial_model_hash = v1.state_digest(model)
    initial_optimizer_hash = optimizer_state_digest(optimizer)
    batch_positions = {
        (int(batch.step), int(batch.microbatch_index)): index
        for index, batch in enumerate(batches)
    }
    timed_records: List[Dict[str, Any]] = []
    for phase, count in (("warmup", warmup_steps), ("timed", timed_steps)):
        for local_step in range(count):
            current_step = local_step + 1
            group_batches = [batch for batch in batches if batch.step == current_step]
            if not group_batches:
                raise ValueError(f"missing schedule step {current_step}")
            record = run_disposable_step(model, optimizer, raw, group_batches, batch_positions, device, hp)
            record["step"] = current_step
            record["phase"] = phase
            if phase == "timed":
                timed_records.append(record)
    torch.cuda.synchronize(device)
    final_model_hash = v1.state_digest(model)
    final_optimizer_hash = optimizer_state_digest(optimizer)
    phase_sums = {
        key: sum(int(record[key]) for record in timed_records)
        for key in (
            "cpu_batch_prepare_ns", "h2d_ns", "forward_host_ns",
            "forward_cuda_event_ns", "backward_host_ns", "backward_cuda_event_ns",
            "optimizer_ns", "wall_ns",
        )
    }
    result = {
        "context": root.name,
        "seed": seed,
        "variant": variant,
        "repetition": repetition,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "initial_model_state_sha256": initial_model_hash,
        "final_model_state_sha256": final_model_hash,
        "initial_optimizer_state_sha256": initial_optimizer_hash,
        "final_optimizer_state_sha256": final_optimizer_hash,
        "timed_records": timed_records,
        "phase_sums": phase_sums,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "all_finite_losses": all(math.isfinite(value) for record in timed_records for value in record["loss_values"]),
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required by the frozen E16 contract")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def run_cell(context_root: Path, seed: int, output_root: Path, warmup: int, timed: int, repetitions: int) -> Dict[str, Any]:
    config_path = context_root / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    if int(hp["num_experts"]) != 16 or int(hp["top_k"]) != 2 or config["contract"]["architectural_branches"] != ["B"]:
        raise ValueError("frozen E16 Branch-B geometry mismatch")
    anchor_path = context_root / "crucible_manifest" / config["anchors"][str(seed)]["relative_path"]
    anchor_sha = v1.sha256_file(anchor_path)
    if anchor_sha != config["anchors"][str(seed)]["binary_payload_sha256"]:
        raise ValueError("anchor SHA mismatch")
    max_steps = warmup + timed
    batches, schedule_report, raw, source_report, binary_report = v1.build_batches(config, context_root, seed, max_steps)
    input_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_report = v1.validate_prefix_inputs(raw, batches, input_tensors, min(32, len(batches)))
    equivalence = {}
    for variant in STRICT_CANDIDATES:
        equivalence[variant] = equivalence_probe(config, context_root, seed, variant, torch.device("cuda"), raw, batches, anchor_path)
        print(json.dumps({"cell": context_root.name, "seed": seed, "equivalence": variant, "accepted": equivalence[variant]["accepted"]}, sort_keys=True), flush=True)
    runs: List[Dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for variant in FULL_VARIANTS:
            result = run_trial(config, context_root, seed, variant, torch.device("cuda"), raw, batches, anchor_path, warmup, timed, repetition)
            runs.append(result)
            print(json.dumps({"cell": context_root.name, "seed": seed, "variant": variant, "rep": repetition, "wall_s": round(result["phase_sums"]["wall_ns"] / 1e9, 4)}, sort_keys=True), flush=True)
    baseline = {(item["variant"], item["repetition"]): item for item in runs}
    state_comparisons = {}
    for repetition in range(1, repetitions + 1):
        ref = baseline[("original", repetition)]
        for variant in STRICT_CANDIDATES:
            cand = baseline[(variant, repetition)]
            state_comparisons[variant] = state_comparisons.get(variant, [])
            state_comparisons[variant].append({
                "repetition": repetition,
                "model_state_equal": ref["final_model_state_sha256"] == cand["final_model_state_sha256"],
                "optimizer_state_equal": ref["final_optimizer_state_sha256"] == cand["final_optimizer_state_sha256"],
            })
    return {
        "context": context_root.name,
        "seed": seed,
        "config_sha256": v1.sha256_file(config_path),
        "anchor_sha256": anchor_sha,
        "source_report": source_report,
        "binary_report": binary_report,
        "schedule_report": schedule_report,
        "input_report": input_report,
        "parameters": {
            "model_parameter_count": int(hp["expected_parameter_geometry"]),
            "context": int(hp["context_window_length"]),
            "num_experts": int(hp["num_experts"]),
            "top_k": int(hp["top_k"]),
            "warmup_steps": warmup,
            "timed_steps": timed,
            "repetitions": repetitions,
            "equivalence_tolerances": {"forward_abs": FORWARD_TOL, "loss_abs": LOSS_TOL, "gradient_abs": GRAD_TOL, "parameter_abs": PARAM_TOL},
        },
        "equivalence": equivalence,
        "state_comparisons_to_original": state_comparisons,
        "runs": runs,
        "integrity": {
            "input_prefix_exact": input_report["exact_equivalence"],
            "strict_equivalence_probes_pass": all(item["accepted"] for item in equivalence.values()),
            "strict_final_state_comparisons_pass": all(
                item["model_state_equal"] and item["optimizer_state_equal"]
                for values in state_comparisons.values() for item in values
            ),
            "all_runs_finite": all(item["all_finite_losses"] for item in runs),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU Neural SSRE v2 strategy landscape")
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", choices=CONTEXT_NAMES, default="context_128")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_cuda()
    device = torch.device("cuda")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        contexts = (args.context,)
        seeds = (args.seed,)
        warmup, timed, repetitions = 1, 2, 1
    else:
        contexts = CONTEXT_NAMES
        seeds = SEEDS
        warmup, timed, repetitions = 2, 8, 2
    cells = []
    for context_name in contexts:
        context_root = Path("/home/timbushnell/crucible_context_gate_v1_20261018") / context_name
        for seed in seeds:
            cells.append(run_cell(context_root, seed, output, warmup, timed, repetitions))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "v1_source_sha256": v1.sha256_file(V1_DIR / "grouped_expert_benchmark.py"),
        "v2_source_sha256": v1.sha256_file(Path(__file__).resolve()),
        "matrix": {"contexts": list(contexts), "seeds": list(seeds), "variants": list(FULL_VARIANTS), "warmup_steps": warmup, "timed_steps": timed, "repetitions": repetitions},
        "cells": cells,
        "integrity": {
            "cell_count": len(cells),
            "all_input_prefixes_exact": all(cell["integrity"]["input_prefix_exact"] for cell in cells),
            "all_strict_probes_pass": all(cell["integrity"]["strict_equivalence_probes_pass"] for cell in cells),
            "all_strict_final_state_comparisons_pass": all(cell["integrity"]["strict_final_state_comparisons_pass"] for cell in cells),
            "all_runs_finite": all(cell["integrity"]["all_runs_finite"] for cell in cells),
            "scientific_training_run": False,
        },
        "limitations": [
            "This is a disposable local execution benchmark, not scientific training.",
            "The context/seed matrix holds the frozen model geometry at E16; expert-count fragmentation is measured separately in the dispatch fixture.",
            "Optimizer-state hashes are now included; matching optimizer configuration alone is not treated as state equivalence.",
            "No neural scheduler was trained or evaluated.",
        ],
    }
    out_path = output / "strategy_landscape_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "strategy_landscape_summary.json": v1.sha256_file(out_path),
        "v2_source_sha256": summary["v2_source_sha256"],
        "v1_source_sha256": summary["v1_source_sha256"],
    }
    (output / "strategy_landscape_output_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "mode": args.mode, "cells": len(cells), "all_strict_pass": summary["integrity"]["all_strict_probes_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
