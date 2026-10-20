#!/usr/bin/env python3
"""Read-only GPU Neural SSRE v0 profile for the frozen Crucible E16 path.

This probe executes one warmed and one profiled forward/backward workload using
the existing frozen model and schedule. It never performs an optimizer update,
never writes to the Crucible package and never changes a checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))

from gpu_step_benchmark import (  # noqa: E402
    assemble_current,
    build_batches,
    load_model,
    loss_sums,
    sha256_file,
    state_digest,
)


SCHEMA_VERSION = "gpu-neural-ssre-v0-profile-20261018"
PAD_ID = 0
RAW_BYTE_START = 3


def now_ns() -> int:
    return time.perf_counter_ns()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def event_name(event: Any) -> str:
    value = getattr(event, "name", None)
    if value is None:
        value = getattr(event, "key", "")
    return str(value)


def event_time_us(event: Any, self_only: bool) -> float:
    """Read PyTorch 2.x device time across CUDA/device naming variants."""

    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if self_only
        else ("device_time_total", "cuda_time_total")
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def cpu_time_us(event: Any, self_only: bool) -> float:
    name = "self_cpu_time_total" if self_only else "cpu_time_total"
    value = getattr(event, name, 0.0)
    return float(value or 0.0)


def sha256_json(path: Path) -> str:
    return sha256_file(path)


def install_profile_ranges(model: torch.nn.Module, expert_records: MutableMapping[str, Dict[str, Any]]) -> None:
    """Add disposable profiler ranges around existing module calls."""

    for layer_index, block in enumerate(model.layers):
        moe = block.moe
        original_moe_forward = moe.forward

        def wrapped_moe_forward(*args: Any, _original=original_moe_forward, _layer=layer_index, **kwargs: Any):
            with torch.profiler.record_function(f"moe/layer_{_layer}"):
                return _original(*args, **kwargs)

        moe.forward = wrapped_moe_forward  # type: ignore[method-assign]

        attention = block.attn
        original_attention_forward = attention.forward

        def wrapped_attention_forward(
            *args: Any,
            _original=original_attention_forward,
            _layer=layer_index,
            **kwargs: Any,
        ):
            with torch.profiler.record_function(f"attention/layer_{_layer}"):
                return _original(*args, **kwargs)

        attention.forward = wrapped_attention_forward  # type: ignore[method-assign]

        for expert_index, expert in enumerate(moe.experts):
            key = f"layer_{layer_index}/expert_{expert_index}"
            expert_records[key] = {
                "layer": layer_index,
                "expert": expert_index,
                "calls": 0,
                "selected_tokens": 0,
                "tokens_per_call": [],
                "input_shapes": [],
                "input_elements": 0,
            }
            original_expert_forward = expert.forward

            def wrapped_expert_forward(
                x: torch.Tensor,
                _original=original_expert_forward,
                _layer=layer_index,
                _expert=expert_index,
                _key=key,
            ):
                record = expert_records[_key]
                token_count = int(x.shape[0])
                record["calls"] += 1
                record["selected_tokens"] += token_count
                record["tokens_per_call"].append(token_count)
                shape = [int(value) for value in x.shape]
                if shape not in record["input_shapes"]:
                    record["input_shapes"].append(shape)
                record["input_elements"] += int(x.numel())
                with torch.profiler.record_function(
                    f"expert/layer_{_layer}/expert_{_expert}"
                ):
                    return _original(x)

            expert.forward = wrapped_expert_forward  # type: ignore[method-assign]


def execute_step(
    model: torch.nn.Module,
    raw: torch.Tensor,
    group_batches: Sequence[Any],
    device: torch.device,
    hp: Mapping[str, Any],
    profile: Any = None,
) -> Dict[str, Any]:
    """Execute a complete schedule step without an optimizer update."""

    model.zero_grad(set_to_none=True)
    total_target_tokens = sum(int(item.nonpad_target_tokens) for item in group_batches)
    losses = []
    valid_targets = 0
    raw_targets = 0
    host_start = now_ns()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    event_start.record()

    for batch_index, batch in enumerate(group_batches):
        x_cpu, y_cpu = assemble_current(raw, batch)
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
        current_sequence_ids = torch.arange(x_gpu.size(0), device=device) + (
            batch_index * int(hp["microbatch_size"])
        )
        range_name = f"microbatch/{batch_index}"
        range_context = torch.profiler.record_function(range_name) if profile is not None else None
        if range_context is not None:
            range_context.__enter__()
        try:
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
                ) / total_target_tokens
            if profile is not None:
                backward_context = torch.profiler.record_function("step/backward")
                backward_context.__enter__()
            else:
                backward_context = None
            try:
                loss.backward()
            finally:
                if backward_context is not None:
                    backward_context.__exit__(None, None, None)
            losses.append(float(loss.detach().item()))
            valid_targets += int(valid_count)
            raw_targets += int(raw_count)
        finally:
            if range_context is not None:
                range_context.__exit__(None, None, None)
        del logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids
    event_end.record()
    event_end.synchronize()
    torch.cuda.synchronize(device)
    host_end = now_ns()
    return {
        "host_wall_ns": host_end - host_start,
        "cuda_event_ns": int(event_start.elapsed_time(event_end) * 1_000_000.0),
        "losses": losses,
        "valid_targets": valid_targets,
        "raw_targets": raw_targets,
    }


def aggregate_named_ranges(events: Sequence[Any], prefix: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for event in events:
        name = event_name(event)
        if not name.startswith(prefix):
            continue
        item = result.setdefault(
            name,
            {"calls": 0, "cpu_us": 0.0, "device_us": 0.0},
        )
        item["calls"] += 1
        item["cpu_us"] += cpu_time_us(event, self_only=False)
        item["device_us"] += event_time_us(event, self_only=False)
    for item in result.values():
        item["cpu_us"] = round(item["cpu_us"], 6)
        item["device_us"] = round(item["device_us"], 6)
    return result


def operator_summary(profiler: Any) -> Tuple[list[Dict[str, Any]], Dict[str, float]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for event in profiler.key_averages(group_by_input_shape=True):
        name = event_name(event)
        item = grouped.setdefault(
            name,
            {"name": name, "calls": 0, "self_cpu_us": 0.0, "self_device_us": 0.0},
        )
        item["calls"] += int(getattr(event, "count", 0) or 0)
        item["self_cpu_us"] += cpu_time_us(event, self_only=True)
        item["self_device_us"] += event_time_us(event, self_only=True)

    rows = sorted(
        grouped.values(),
        key=lambda item: item["self_device_us"],
        reverse=True,
    )
    for item in rows:
        item["self_cpu_us"] = round(item["self_cpu_us"], 6)
        item["self_device_us"] = round(item["self_device_us"], 6)

    patterns = {
        "matrix_or_linear": ("mm", "addmm", "bmm", "matmul", "linear"),
        "routing_or_indexing": (
            "topk",
            "gather",
            "scatter",
            "index",
            "bincount",
            "put",
        ),
        "layout_or_copy": ("copy", "contiguous", "to_copy", "_to_"),
        "activation_or_elementwise": ("silu", "mul", "add", "sub", "div"),
    }
    category_totals = {name: 0.0 for name in patterns}
    for item in rows:
        lowered = item["name"].lower()
        for category, needles in patterns.items():
            if any(needle in lowered for needle in needles):
                category_totals[category] += item["self_device_us"]
    return rows[:80], {key: round(value, 6) for key, value in category_totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only frozen E16 GPU MoE profile")
    parser.add_argument(
        "--context-root",
        type=Path,
        default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required by the frozen E16 path")

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

    batches, schedule_report, raw, source_report, binary_report = build_batches(
        config, root, args.seed, args.max_steps
    )
    first_step = min(batch.step for batch in batches)
    group_batches = [batch for batch in batches if batch.step == first_step]

    model = load_model(anchor_path, hp, device)
    initial_state = state_digest(model)

    # One unprofiled pass warms CUDA kernels. No optimizer update is performed.
    warmup = execute_step(model, raw, group_batches, device, hp)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    expert_records: Dict[str, Dict[str, Any]] = {}
    install_profile_ranges(model, expert_records)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        profiled = execute_step(model, raw, group_batches, device, hp, profile=profiler)

    final_state = state_digest(model)
    events = profiler.events()
    named_ranges = {
        "moe": aggregate_named_ranges(events, "moe/"),
        "expert": aggregate_named_ranges(events, "expert/"),
        "attention": aggregate_named_ranges(events, "attention/"),
        "step": aggregate_named_ranges(events, "step/"),
    }
    top_operators, category_totals = operator_summary(profiler)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    for record in expert_records.values():
        record["mean_tokens_per_call"] = (
            record["selected_tokens"] / record["calls"] if record["calls"] else 0.0
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "read-only GPU-side E16 MoE attribution profile",
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "context_root": str(root),
        "config_sha256": sha256_file(config_path),
        "anchor_sha256": anchor_sha,
        "source_report": source_report,
        "binary_report": binary_report,
        "schedule_report": schedule_report,
        "parameters": {
            "seed": args.seed,
            "profiled_step": first_step,
            "schedule_group_microbatches": len(group_batches),
            "model_parameter_count": int(hp["expected_parameter_geometry"]),
            "context": int(hp["context_window_length"]),
            "num_experts": int(hp["num_experts"]),
            "top_k": int(hp["top_k"]),
            "d_model": int(hp["d_model"]),
            "d_ff": int(hp["d_ff"]),
            "n_layers": int(hp["n_layers"]),
        },
        "warmup": warmup,
        "profiled_step": profiled,
        "expert_dispatch_records": expert_records,
        "profiler_ranges": named_ranges,
        "top_cuda_operators_by_self_device_time": top_operators,
        "overlapping_operator_category_totals_us": category_totals,
        "memory": {
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        },
        "integrity": {
            "all_profiled_losses_finite": all(finite_number(value) for value in profiled["losses"]),
            "initial_state_sha256": initial_state,
            "final_state_sha256": final_state,
            "model_state_unchanged": initial_state == final_state,
            "optimizer_step_performed": False,
            "frozen_inputs_consumed": True,
        },
        "interpretation_limits": [
            "Nested MoE, expert and attention ranges overlap and are not additive.",
            "Operator category totals overlap and are not additive.",
            "This is a one-step attribution profile, not a throughput benchmark.",
            "No grouped-expert implementation or neural scheduler was evaluated.",
        ],
    }
    out_path = output / "gpu_moe_profile_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_hash = sha256_file(out_path)
    (output / "profile_output_hashes.json").write_text(
        json.dumps(
            {
                "gpu_moe_profile_summary.json": summary_hash,
                "frozen_config_sha256": summary["config_sha256"],
                "frozen_anchor_sha256": anchor_sha,
                "profile_source_sha256": sha256_file(Path(__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(out_path),
                "device": summary["device"],
                "model_state_unchanged": summary["integrity"]["model_state_unchanged"],
                "expert_ranges": len(named_ranges["expert"]),
                "moe_ranges": len(named_ranges["moe"]),
                "profiled_cuda_event_ms": round(profiled["cuda_event_ns"] / 1e6, 3),
                "peak_vram_gib": round(peak_allocated / (1024**3), 3),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

