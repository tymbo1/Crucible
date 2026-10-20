#!/usr/bin/env python3
"""Short synchronized local-GPU benchmark for the frozen Crucible E16 path.

This is an engineering benchmark, not a scientific training run. Each arm
loads a fresh disposable model from an immutable E16 anchor, consumes the same
frozen schedule prefix, and performs a small number of warmup/timed optimizer
steps. No checkpoint, manifest or prior result is written.

The benchmark separates CPU assembly, H2D transfer, forward, backward,
optimizer and telemetry-on/off effects. CUDA event timings are paired with
host timings so Python launch/synchronization overhead is visible as the
difference between the two.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


V3_DIR = Path("/home/timbushnell/ssre_npa_v3_measured_batch_replay_20261018")
GROUPED_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
CONTEXT_RUNNER_DIR = Path("/home/timbushnell/crucible_v2_production_telemetry_v1_20261019/shared")
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(GROUPED_DIR))
sys.path.insert(0, str(CONTEXT_RUNNER_DIR))

from cpu_replay import (  # noqa: E402
    Batch,
    batch_fingerprint,
    load_binary,
    load_manifest,
    load_schedule,
    load_source_records,
    sha256_bytes,
    sha256_file,
)
from model_core import BraidPrototypeLM, ResearchMoETelemetry  # noqa: E402
import grouped_expert_benchmark as grouped  # noqa: E402


PAD_ID = 0
RAW_BYTE_START = 3
SCHEMA_VERSION = "crucible-v2-production-telemetry-overhead-benchmark-20261019"


def now_ns() -> int:
    return time.perf_counter_ns()


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(anchor_path: Path, hp: Mapping[str, Any], device: torch.device) -> BraidPrototypeLM:
    model = BraidPrototypeLM(
        int(hp["vocab_size"]),
        int(hp["n_layers"]),
        int(hp["d_model"]),
        int(hp["n_heads"]),
        int(hp["latent_dim_kv"]),
        int(hp["latent_dim_q"]),
        int(hp["pe_dim"]),
        int(hp["num_experts"]),
        int(hp["d_ff"]),
        top_k=int(hp["top_k"]),
    )
    expected = int(hp["expected_parameter_geometry"])
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != expected:
        raise ValueError(f"model geometry mismatch: {actual} != {expected}")
    state = torch.load(anchor_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    model.train()
    return model


def make_optimizer(model: nn.Module, hp: Mapping[str, Any], lr: float) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    no_decay_types = (nn.RMSNorm, nn.LayerNorm, nn.Embedding)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            (no_decay if isinstance(module, no_decay_types) or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(hp["weight_decay"]), "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ],
        betas=(float(hp["adamw_beta1"]), float(hp["adamw_beta2"])),
        eps=float(hp["adamw_eps"]),
    )


def loss_sums(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    values = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    flat = targets.reshape(-1)
    valid = flat != PAD_ID
    raw = flat >= RAW_BYTE_START
    return values[valid].sum(), values[raw].sum(), int(valid.sum().item()), int(raw.sum().item())


def cuda_timed(device: torch.device, fn: Any) -> Tuple[Any, int, int]:
    """Return result, synchronized host wall ns and CUDA event ns."""
    torch.cuda.synchronize(device)
    host_start = now_ns()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    result = fn()
    event_end.record()
    event_end.synchronize()
    host_end = now_ns()
    event_ns = int(event_start.elapsed_time(event_end) * 1_000_000.0)
    return result, host_end - host_start, event_ns


def assemble_current(raw: torch.Tensor, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([raw[index, 0].clone() for index in batch.indices]),
        torch.stack([raw[index, 1].clone() for index in batch.indices]),
    )


def assemble_indexed(raw: torch.Tensor, batch: Batch, index_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = raw.index_select(0, index_tensor)
    return selected[:, 0].contiguous(), selected[:, 1].contiguous()


def validate_prefix_inputs(
    raw: torch.Tensor,
    batches: Sequence[Batch],
    index_tensors: Sequence[torch.Tensor],
    prefix_batches: int,
) -> Dict[str, Any]:
    reference: List[str] = []
    candidate: List[str] = []
    for batch, index_tensor in zip(batches[:prefix_batches], index_tensors[:prefix_batches]):
        x_a, y_a = assemble_current(raw, batch)
        x_b, y_b = assemble_indexed(raw, batch, index_tensor)
        fp_a = batch_fingerprint(x_a, y_a)
        fp_b = batch_fingerprint(x_b, y_b)
        reference.append(fp_a["fingerprint_sha256"])
        candidate.append(fp_b["fingerprint_sha256"])
        if fp_a != fp_b:
            raise ValueError(f"CPU input mismatch at batch {batch.step}/{batch.microbatch_index}")
        if int(fp_a["valid_target_count"]) != batch.nonpad_target_tokens:
            raise ValueError("valid target count mismatch")
        if int(fp_a["raw_target_count"]) != batch.raw_byte_target_tokens:
            raise ValueError("raw target count mismatch")
    plan_payload = "|".join(reference).encode("ascii")
    return {
        "prefix_batches": prefix_batches,
        "reference_plan_sha256": sha256_bytes(plan_payload),
        "candidate_plan_sha256": sha256_bytes("|".join(candidate).encode("ascii")),
        "exact_equivalence": reference == candidate,
    }


def build_batches(config: Mapping[str, Any], root: Path, seed: int, max_steps: int) -> Tuple[List[Batch], Dict[str, Any], torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    manifest_base = root / "crucible_manifest"
    train_info = config["splits"]["train"]
    records, source_report = load_source_records(manifest_base, train_info)
    raw, binary_report = load_binary(manifest_base, train_info, len(records), int(config["hyperparameters"]["context_window_length"]))
    schedule_info = config["schedules"][str(seed)]
    batches, schedule_report = load_schedule(manifest_base, schedule_info, seed, [len(item) for item in records])
    selected = [batch for batch in batches if batch.step <= max_steps]
    if not selected:
        raise ValueError("empty benchmark prefix")
    return selected, schedule_report, raw, source_report, binary_report


def run_trial(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    root: Path,
    seed: int,
    arm: str,
    telemetry: bool,
    repetition: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    raw: torch.Tensor,
    batches: Sequence[Batch],
    index_tensors: Sequence[torch.Tensor],
    anchor_path: Path,
) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    model = load_model(anchor_path, hp, device)
    if arm not in ("original", "grouped_rank_preserving"):
        raise ValueError(f"unknown execution variant: {arm}")
    grouped.patch_grouped_model(model, arm)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = make_optimizer(model, hp, float(config["contract"]["learning_rate_candidates"][0]))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_state_sha = state_digest(model)
    ResearchMoETelemetry.configure(int(hp["vocab_size"]))
    ResearchMoETelemetry.reset_step_buffer()
    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    max_steps = warmup_steps + timed_steps
    if len(batches) == 0:
        raise ValueError("no batches")
    timed_records: List[Dict[str, Any]] = []
    total_start = now_ns()
    total_cpu_start = time.process_time_ns()
    step_cursor = 0
    timed_batch_cursor = 0
    warmup_batch_cursor = 0

    for phase, count in (("warmup", warmup_steps), ("timed", timed_steps)):
        for local_step in range(count):
            current_step = (warmup_steps if phase == "timed" else 0) + local_step + 1
            group_batches = [batch for batch in batches if batch.step == current_step]
            if not group_batches:
                raise ValueError(f"missing step {current_step} in benchmark prefix")
            optimizer.zero_grad()
            step_start = now_ns()
            step_record = {
                "step": current_step,
                "phase": phase,
                "microbatches": len(group_batches),
                "cpu_batch_prepare_ns": 0,
                "h2d_ns": 0,
                "forward_host_ns": 0,
                "forward_cuda_event_ns": 0,
                "backward_host_ns": 0,
                "backward_cuda_event_ns": 0,
                "optimizer_ns": 0,
                "loss_values": [],
                "grad_norm": None,
                "valid_targets": 0,
                "raw_targets": 0,
                "source_bytes": 0,
            }
            for batch_index_in_step, batch in enumerate(group_batches):
                batch_global_index = next((index for index, item in enumerate(batches) if item is batch), None)
                if batch_global_index is None:
                    raise ValueError("batch identity lookup failed")
                prepare_start = now_ns()
                if arm in ("original", "grouped_rank_preserving"):
                    x_cpu, y_cpu = assemble_current(raw, batch)
                elif arm == "indexed_precomputed":
                    x_cpu, y_cpu = assemble_indexed(raw, batch, index_tensors[batch_global_index])
                else:
                    raise ValueError(f"unknown arm: {arm}")
                prepare_end = now_ns()
                step_record["cpu_batch_prepare_ns"] += prepare_end - prepare_start
                h2d_start = now_ns()
                torch.cuda.synchronize(device)
                x_gpu = x_cpu.to(device)
                y_gpu = y_cpu.to(device)
                torch.cuda.synchronize(device)
                step_record["h2d_ns"] += now_ns() - h2d_start
                seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
                current_sequence_ids = torch.arange(x_gpu.size(0), device=device) + (
                    batch_index_in_step * int(hp["microbatch_size"])
                )

                def forward_call() -> Tuple[torch.Tensor, torch.Tensor, int, int]:
                    with autocast_context:
                        logits = model(
                            x_gpu,
                            seq_idx,
                            current_branch="B",
                            current_sequence_ids=current_sequence_ids,
                            telemetry_class=ResearchMoETelemetry if telemetry else None,
                        )
                        ce_sum, raw_ce_sum, valid_count, raw_count = loss_sums(logits, y_gpu)
                        auxiliary_loss = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
                        loss = (ce_sum + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count) / int(group_batches[0].nonpad_target_tokens + sum(item.nonpad_target_tokens for item in group_batches[1:]))
                    return loss, logits, valid_count, raw_count

                (loss, _logits, valid_count, raw_count), forward_host, forward_cuda = cuda_timed(device, forward_call)
                step_record["forward_host_ns"] += forward_host
                step_record["forward_cuda_event_ns"] += forward_cuda
                step_record["valid_targets"] += valid_count
                step_record["raw_targets"] += raw_count
                step_record["source_bytes"] += batch.source_bytes
                step_record["loss_values"].append(float(loss.detach().item()))

                def backward_call() -> None:
                    loss.backward()

                _, backward_host, backward_cuda = cuda_timed(device, backward_call)
                step_record["backward_host_ns"] += backward_host
                step_record["backward_cuda_event_ns"] += backward_cuda
                del _logits, loss, x_gpu, y_gpu, seq_idx, current_sequence_ids
            clip_start = now_ns()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(hp["max_norm_clip"])).item())
            clip_ns = now_ns() - clip_start
            optimizer_start = now_ns()
            torch.cuda.synchronize(device)
            optimizer.step()
            torch.cuda.synchronize(device)
            step_record["optimizer_ns"] = now_ns() - optimizer_start + clip_ns
            step_record["grad_norm"] = grad_norm
            step_record["wall_ns"] = now_ns() - step_start
            step_record["non_compute_ns_proxy"] = max(
                0,
                step_record["wall_ns"]
                - step_record["forward_cuda_event_ns"]
                - step_record["backward_cuda_event_ns"],
            )
            if phase == "timed":
                timed_records.append(step_record)
            step_cursor += 1
    telemetry_finalize_ns = 0
    if telemetry:
        torch.cuda.synchronize(device)
        finalize_start = now_ns()
        _summary = ResearchMoETelemetry.get_summary("B")
        ResearchMoETelemetry.reset_step_buffer()
        torch.cuda.synchronize(device)
        telemetry_finalize_ns = now_ns() - finalize_start
    else:
        ResearchMoETelemetry.reset_step_buffer()
    total_ns = now_ns() - total_start
    cpu_ns = time.process_time_ns() - total_cpu_start
    final_state_sha = state_digest(model)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    phase_sums = {
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
            "non_compute_ns_proxy",
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "arm": arm,
        "telemetry": telemetry,
        "repetition": repetition,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "parameter_count": parameter_count,
        "config_sha256": config_sha,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "total_wall_ns": total_ns,
        "process_cpu_ns": cpu_ns,
        "cpu_time_over_wall": cpu_ns / max(1, total_ns),
        "telemetry_finalize_ns": telemetry_finalize_ns,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "phase_sums": phase_sums,
        "timed_records": timed_records,
        "numerical": {
            "all_finite_losses": all(math.isfinite(value) for record in timed_records for value in record["loss_values"]),
            "all_finite_grad_norms": all(record["grad_norm"] is not None and math.isfinite(record["grad_norm"]) for record in timed_records),
            "total_valid_targets": sum(record["valid_targets"] for record in timed_records),
            "total_raw_targets": sum(record["raw_targets"] for record in timed_records),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Short synchronized local GPU E16 benchmark")
    parser.add_argument("--context-root", type=Path, default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this benchmark")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required by the frozen E16 contract")
    if args.deterministic:
        # The production telemetry package has the two weighted CUDA
        # reductions on CPU, so strict deterministic mode is appropriate.
        torch.use_deterministic_algorithms(True, warn_only=False)
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
    config, config_sha = load_manifest(root)
    hp = config["hyperparameters"]
    if int(hp["num_experts"]) != 16 or int(hp["top_k"]) != 2 or config["contract"]["architectural_branches"] != ["B"]:
        raise ValueError("frozen E16 Branch-B geometry mismatch")
    anchor_path = root / "crucible_manifest" / config["anchors"][str(args.seed)]["relative_path"]
    anchor_sha = sha256_file(anchor_path)
    expected_anchor_sha = config["anchors"][str(args.seed)]["binary_payload_sha256"]
    if anchor_sha != expected_anchor_sha:
        raise ValueError("anchor SHA mismatch")
    max_steps = args.warmup_steps + args.timed_steps
    batches, schedule_report, raw, source_report, binary_report = build_batches(config, root, args.seed, max_steps)
    index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_report = validate_prefix_inputs(raw, batches, index_tensors, min(32, len(batches)))
    torch.cuda.reset_peak_memory_stats(device)
    runs: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for arm in ("original", "grouped_rank_preserving"):
            for telemetry in (False, True):
                torch.manual_seed(12345)
                result = run_trial(
                    config=config,
                    config_sha=config_sha,
                    root=root,
                    seed=args.seed,
                    arm=arm,
                    telemetry=telemetry,
                    repetition=repetition,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    device=device,
                    raw=raw,
                    batches=batches,
                    index_tensors=index_tensors,
                    anchor_path=anchor_path,
                )
                runs.append(result)
                print(json.dumps({
                    "arm": arm,
                    "telemetry": telemetry,
                    "repetition": repetition,
                    "wall_s": round(result["total_wall_ns"] / 1e9, 3),
                    "peak_vram_gib": round(result["peak_cuda_allocated_bytes"] / (1024 ** 3), 3),
                    "final_state": result["final_state_sha256"][:16],
                }, sort_keys=True), flush=True)
                del result
                gc.collect()
                torch.cuda.empty_cache()
    equivalence: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for telemetry in (False, True):
            baseline = next(item for item in runs if item["arm"] == "original" and item["telemetry"] == telemetry and item["repetition"] == repetition)
            indexed = next(item for item in runs if item["arm"] == "grouped_rank_preserving" and item["telemetry"] == telemetry and item["repetition"] == repetition)
            baseline_losses = [value for record in baseline["timed_records"] for value in record["loss_values"]]
            indexed_losses = [value for record in indexed["timed_records"] for value in record["loss_values"]]
            max_loss_delta = max((abs(a - b) for a, b in zip(baseline_losses, indexed_losses)), default=0.0)
            equivalence.append({
                "repetition": repetition,
                "telemetry": telemetry,
                "final_state_sha_equal": baseline["final_state_sha256"] == indexed["final_state_sha256"],
                "max_loss_abs_delta": max_loss_delta,
                "loss_tolerance": 1e-5,
                "losses_within_tolerance": max_loss_delta <= 1e-5,
                "input_prefix_exact": input_report["exact_equivalence"],
            })
    summary = {
        "schema_version": "crucible-v2-production-telemetry-overhead-summary-20261019",
        "deterministic_mode": args.deterministic,
        "device": torch.cuda.get_device_name(device),
        "device_index": torch.cuda.current_device(),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "context_root": str(root),
        "config_sha256": config_sha,
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
            "execution_variants": ["original", "grouped_rank_preserving"],
            "telemetry_modes": [False, True],
            "model_parameter_count": int(hp["expected_parameter_geometry"]),
            "context": int(hp["context_window_length"]),
            "num_experts": int(hp["num_experts"]),
            "top_k": int(hp["top_k"]),
        },
        "runs": runs,
        "equivalence": equivalence,
        "integrity": {
            "all_runs_finite": all(item["numerical"]["all_finite_losses"] and item["numerical"]["all_finite_grad_norms"] for item in runs),
            "all_input_prefixes_exact": input_report["exact_equivalence"],
            "all_loss_tolerances_pass": all(item["losses_within_tolerance"] for item in equivalence),
            "all_final_state_hashes_equal": all(item["final_state_sha_equal"] for item in equivalence),
        },
        "measurement_limitations": [
            "This is a short disposable local benchmark, not a scientific training run.",
            "Host-to-device timings use synchronous .to(device), matching the current runner; pinned/nonblocking staging is not tested here.",
            "Forward/backward CUDA event time is paired with host wall time; their difference is a launch/synchronization proxy, not a hardware counter.",
            "GPU idle time is reported as a non-compute proxy, not sampled utilization.",
        ],
    }
    (output / "gpu_step_benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "device": summary["device"],
        "runs": len(runs),
        "all_finite": summary["integrity"]["all_runs_finite"],
        "equivalence": summary["integrity"]["all_final_state_hashes_equal"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

