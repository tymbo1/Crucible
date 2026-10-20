#!/usr/bin/env python3
"""SSRE-NPA v5 isolated telemetry-efficiency benchmark.

The frozen Crucible model and model_core.py are imported read-only.  This
script compares the existing ResearchMoETelemetry class with disposable
telemetry backends that keep the same scientific counters but reduce Python
loops and host-transfer fragmentation.  It also records one torch-profiler
trace summary for attention/MoE/backward/optimizer attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
sys.path.insert(0, str(V4_DIR))

from gpu_step_benchmark import (  # noqa: E402
    assemble_current,
    build_batches,
    cuda_timed,
    load_model,
    make_optimizer,
    state_digest,
    validate_prefix_inputs,
)
from model_core import BraidPrototypeLM, ResearchMoETelemetry  # noqa: E402


PAD_ID = 0
RAW_BYTE_START = 3
SCHEMA_VERSION = "ssre-npa-telemetry-efficiency-v5"


def now_ns() -> int:
    return time.perf_counter_ns()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


class BatchedGpuTelemetry:
    """GPU accumulator with one packed host transfer at summary time.

    `diagnostic=True` preserves every field emitted by the frozen telemetry
    implementation.  `diagnostic=False` retains the fields required by the
    preregistered byte-affinity/load evaluator and omits optional per-token
    selection and expert-output diagnostic matrices.
    """

    diagnostic = True
    backend_id = "gpu_full_batched"
    _vocab_size = 259
    _accumulators: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
        "A": {}, "B": {}, "C": {}
    }

    @classmethod
    def configure(cls, vocab_size: int) -> None:
        if int(vocab_size) < 3:
            raise ValueError("Telemetry vocabulary must include PAD/BOS/EOS")
        cls._vocab_size = int(vocab_size)

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
        if layer_id in cls._accumulators[branch]:
            return
        accumulator: Dict[str, torch.Tensor] = {
            "tokens_routed": torch.zeros(1, dtype=torch.long, device=device),
            "hard_assignments": torch.zeros(num_experts, dtype=torch.float32, device=device),
            "entropy_sum": torch.zeros(1, dtype=torch.float32, device=device),
            "token_affinity": torch.zeros(
                num_experts, cls._vocab_size, dtype=torch.float64, device=device
            ),
            "top1_counts": torch.zeros(num_experts, dtype=torch.float64, device=device),
            "top2_counts": torch.zeros(num_experts, dtype=torch.float64, device=device),
        }
        if cls.diagnostic:
            accumulator.update({
                "selected_token_counts": torch.zeros(
                    num_experts, cls._vocab_size, dtype=torch.float64, device=device
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
            })
        cls._accumulators[branch][layer_id] = accumulator

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
            raise ValueError("Routing index shape is inconsistent with probabilities")
        top_k = int(indices.shape[1])
        if valid_tokens.ndim != 1 or valid_tokens.shape[0] != num_tokens:
            raise ValueError("Token ids are inconsistent with routing rows")
        if valid_tokens.min().item() < 0 or valid_tokens.max().item() >= cls._vocab_size:
            raise ValueError("Token id outside telemetry vocabulary")
        cls._init_layer_buffers(branch, layer_id, num_experts, probabilities.device)
        accumulator = cls._accumulators[branch][layer_id]
        accumulator["top_k"] = top_k  # type: ignore[assignment]
        accumulator["tokens_routed"] += num_tokens

        flat_indices = indices.reshape(-1)
        hard_counts = torch.bincount(flat_indices, minlength=num_experts).float()
        accumulator["hard_assignments"] += hard_counts
        accumulator["top2_counts"] += hard_counts.double()
        accumulator["top1_counts"] += torch.bincount(
            indices[:, 0], minlength=num_experts
        ).double()

        if cls.diagnostic:
            normalized_selected = torch.gather(probabilities, 1, indices)
            normalized_selected = normalized_selected / (
                normalized_selected.sum(dim=-1, keepdim=True) + 1e-12
            )
            accumulator["selected_probability_sum"] += torch.bincount(
                flat_indices,
                weights=normalized_selected.reshape(-1).double(),
                minlength=num_experts,
            )
            # This is exactly the same integer count as hard_assignments, but
            # kept in float64 to match the frozen output schema.
            accumulator["selected_probability_count"] += hard_counts.double()
            selected_tokens = valid_tokens.unsqueeze(1).expand(-1, top_k).reshape(-1)
            pair_ids = flat_indices * cls._vocab_size + selected_tokens
            selected_hist = torch.bincount(
                pair_ids,
                minlength=num_experts * cls._vocab_size,
            ).reshape(num_experts, cls._vocab_size).double()
            accumulator["selected_token_counts"] += selected_hist

        accumulator["token_affinity"].index_add_(
            1,
            valid_tokens,
            probabilities.double().transpose(0, 1),
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
        if not cls.diagnostic:
            return
        accumulator = cls._accumulators[branch][layer_id]
        norms = expert_output.detach().float().norm(dim=-1).double()
        accumulator["activation_norm_sum"][expert_index] += norms.sum()
        accumulator["activation_norm_count"][expert_index] += norms.numel()

    @classmethod
    def _fields(cls) -> Sequence[str]:
        fields = (
            "tokens_routed", "entropy_sum", "hard_assignments", "top1_counts",
            "top2_counts", "token_affinity",
        )
        if cls.diagnostic:
            fields += (
                "selected_token_counts", "selected_probability_sum",
                "selected_probability_count", "activation_norm_sum",
                "activation_norm_count",
            )
        return fields

    @classmethod
    def _pack_to_host(cls, branch: str) -> Tuple[List[str], Dict[str, Dict[str, Any]], torch.Tensor]:
        layers = sorted(cls._accumulators[branch])
        metadata: Dict[str, Dict[str, Any]] = {}
        parts: List[torch.Tensor] = []
        for layer in layers:
            accumulator = cls._accumulators[branch][layer]
            for field in cls._fields():
                tensor = accumulator[field].detach()
                metadata[f"{layer}:{field}"] = {
                    "shape": list(tensor.shape),
                    "count": tensor.numel(),
                }
                parts.append(tensor.reshape(-1).to(torch.float64))
        if not parts:
            return layers, metadata, torch.empty(0, dtype=torch.float64)
        # One concatenated device-to-host transfer for all layers and fields.
        return layers, metadata, torch.cat(parts).cpu()

    @classmethod
    def get_summary(cls, branch: str) -> Dict[str, Any]:
        layers, metadata, packed = cls._pack_to_host(branch)
        cursor = 0
        summary: Dict[str, Any] = {}
        fields = cls._fields()
        for layer in layers:
            values: Dict[str, torch.Tensor] = {}
            accumulator = cls._accumulators[branch][layer]
            for field in fields:
                meta = metadata[f"{layer}:{field}"]
                count = int(meta["count"])
                shape = tuple(int(item) for item in meta["shape"])
                values[field] = packed[cursor : cursor + count].reshape(shape)
                cursor += count
            total_tokens = int(round(float(values["tokens_routed"].item())))
            hard = values["hard_assignments"]
            # The frozen implementation derives CV and max:min from its
            # float32 hard-assignment accumulator. Reuse that dtype here so
            # the optimized reporting path is numerically equivalent rather
            # than merely close in float64.
            hard_metrics = hard.to(torch.float32)
            affinity = values["token_affinity"]
            affinity_mass = float(affinity.sum().item())
            max_entropy = math.log(float(hard_metrics.numel()))
            if hard_metrics.numel() == 1:
                global_cv = 0.0
            else:
                global_cv = float((hard_metrics.std(unbiased=True) / (hard_metrics.mean() + 1e-10)).item())
            point: Dict[str, Any] = {
                "top_k": int(accumulator.get("top_k", 0)),
                "tokens_routed": total_tokens,
                "_token_affinity": affinity.tolist(),
                "top1_counts": values["top1_counts"].tolist(),
                "top2_counts": values["top2_counts"].tolist(),
                "hard_assignments": hard.tolist(),
                "affinity_mass_total": affinity_mass,
                "global_hard_cv": global_cv,
                "max_min_hard_ratio": float((hard_metrics.max() / (hard_metrics.min() + 1e-10)).item()),
                "normalized_entropy": (
                    float(values["entropy_sum"].item()) / max(1, total_tokens)
                ) / (max_entropy + 1e-10),
            }
            if cls.diagnostic:
                selected = values["selected_token_counts"]
                selected_sum = values["selected_probability_sum"]
                selected_count = values["selected_probability_count"]
                activation_sum = values["activation_norm_sum"]
                activation_count = values["activation_norm_count"]
                point.update({
                    "selected_token_counts": selected.tolist(),
                    "selected_token_mass_total": float(selected.sum().item()),
                    "selected_probability_sum": selected_sum.tolist(),
                    "selected_probability_count": selected_count.tolist(),
                    "mean_selected_probability": (
                        selected_sum / (selected_count + 1e-12)
                    ).tolist(),
                    "activation_norm_sum": activation_sum.tolist(),
                    "activation_norm_count": activation_count.tolist(),
                    "mean_expert_output_norm": (
                        activation_sum / (activation_count + 1e-12)
                    ).tolist(),
                })
            summary[layer] = point
        if cursor != packed.numel():
            raise AssertionError("telemetry packed-transfer cursor mismatch")
        return summary


class GpuMinimalScientificTelemetry(BatchedGpuTelemetry):
    diagnostic = False
    backend_id = "minimal_scientific"


class BufferedFullTelemetry(BatchedGpuTelemetry):
    diagnostic = True
    backend_id = "buffered_full"


def nested_max_abs(a: Any, b: Any) -> float:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((nested_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return 0.0 if a == b else float("inf")


def compare_telemetry(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    diagnostic: bool,
) -> Dict[str, Any]:
    if set(reference) != set(candidate):
        return {"layers_equal": False, "missing_or_extra_layers": True}
    exact_fields = ["top_k", "tokens_routed", "top1_counts", "top2_counts", "hard_assignments"]
    primary_float_fields = ["_token_affinity", "affinity_mass_total", "global_hard_cv", "max_min_hard_ratio", "normalized_entropy"]
    diagnostic_fields = [
        "selected_token_counts", "selected_token_mass_total", "selected_probability_sum",
        "selected_probability_count", "mean_selected_probability", "activation_norm_sum",
        "activation_norm_count", "mean_expert_output_norm",
    ]
    exact_ok = True
    max_abs = 0.0
    missing_fields: List[str] = []
    for layer in sorted(reference):
        left = reference[layer]
        right = candidate[layer]
        for field in exact_fields:
            if field not in left or field not in right:
                missing_fields.append(f"{layer}.{field}")
                exact_ok = False
            elif left[field] != right[field]:
                exact_ok = False
        fields = primary_float_fields + (diagnostic_fields if diagnostic else [])
        for field in fields:
            if field not in left or field not in right:
                missing_fields.append(f"{layer}.{field}")
                continue
            max_abs = max(max_abs, nested_max_abs(left[field], right[field]))
    return {
        "layers_equal": True,
        "exact_counter_fields_equal": exact_ok and not missing_fields,
        "max_float_abs_delta": max_abs,
        "float_tolerance": 1e-6,
        "float_fields_within_tolerance": max_abs <= 1e-6,
        "all_required_fields_present": not missing_fields,
        "missing_fields": missing_fields,
        "pass": exact_ok and not missing_fields and max_abs <= 1e-6,
    }


def run_trial(
    *,
    backend_name: str,
    telemetry: bool,
    repetition: int,
    config: Mapping[str, Any],
    config_sha: str,
    root: Path,
    raw: torch.Tensor,
    batches: Sequence[Any],
    anchor_path: Path,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    backend_map = {
        "current_full": ResearchMoETelemetry,
        "gpu_full_batched": BatchedGpuTelemetry,
        "minimal_scientific": GpuMinimalScientificTelemetry,
        "buffered_full": BufferedFullTelemetry,
    }
    backend = backend_map.get(backend_name)
    if telemetry and backend is None:
        raise ValueError(f"unknown telemetry backend: {backend_name}")
    model = load_model(anchor_path, hp, device)
    initial_state_sha = state_digest(model)
    optimizer = make_optimizer(model, hp, float(config["contract"]["learning_rate_candidates"][0]))
    if telemetry:
        backend.configure(int(hp["vocab_size"]))
        backend.reset_step_buffer()
    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    groups: Dict[int, List[Any]] = {}
    for batch in batches:
        groups.setdefault(int(batch.step), []).append(batch)
    max_steps = warmup_steps + timed_steps
    losses: List[float] = []
    phase_totals = {
        "cpu_batch_prepare_ns": 0,
        "h2d_ns": 0,
        "forward_host_ns": 0,
        "forward_cuda_event_ns": 0,
        "backward_host_ns": 0,
        "backward_cuda_event_ns": 0,
        "optimizer_ns": 0,
        "wall_ns": 0,
        "non_compute_ns_proxy": 0,
    }
    total_start = now_ns()
    step_records: List[Dict[str, Any]] = []
    for phase, count in (("warmup", warmup_steps), ("timed", timed_steps)):
        for local_step in range(count):
            current_step = local_step + 1 if phase == "warmup" else warmup_steps + local_step + 1
            group_batches = groups.get(current_step)
            if not group_batches:
                raise ValueError(f"missing schedule group {current_step}")
            optimizer.zero_grad()
            step_start = now_ns()
            record = {key: 0 for key in phase_totals}
            record["step"] = current_step
            record["phase"] = phase
            record["loss_values"] = []
            for batch_index, batch in enumerate(group_batches):
                prep_start = now_ns()
                x_cpu, y_cpu = assemble_current(raw, batch)
                record["cpu_batch_prepare_ns"] += now_ns() - prep_start
                h2d_start = now_ns()
                torch.cuda.synchronize(device)
                x_gpu = x_cpu.to(device)
                y_gpu = y_cpu.to(device)
                torch.cuda.synchronize(device)
                record["h2d_ns"] += now_ns() - h2d_start
                seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
                sequence_ids = torch.arange(x_gpu.size(0), device=device) + batch_index * int(hp["microbatch_size"])

                def forward_call() -> Tuple[torch.Tensor, torch.Tensor]:
                    with autocast_context:
                        logits = model(
                            x_gpu,
                            seq_idx,
                            current_branch="B",
                            current_sequence_ids=sequence_ids,
                            telemetry_class=backend if telemetry else None,
                        )
                        ce_sum, _, valid_count, _ = loss_sums(logits, y_gpu)
                        aux = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
                        denom = int(sum(item.nonpad_target_tokens for item in group_batches))
                        loss = (ce_sum + float(hp["branch_b_aux_coef"]) * aux * valid_count) / denom
                    return loss, logits

                (loss, logits), host_ns, cuda_ns = cuda_timed(device, forward_call)
                record["forward_host_ns"] += host_ns
                record["forward_cuda_event_ns"] += cuda_ns
                record["loss_values"].append(float(loss.detach().item()))

                def backward_call() -> None:
                    loss.backward()

                _, host_ns, cuda_ns = cuda_timed(device, backward_call)
                record["backward_host_ns"] += host_ns
                record["backward_cuda_event_ns"] += cuda_ns
                del logits, loss, x_gpu, y_gpu, seq_idx, sequence_ids
            clip_start = now_ns()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["max_norm_clip"]))
            optimizer_start = now_ns()
            torch.cuda.synchronize(device)
            optimizer.step()
            torch.cuda.synchronize(device)
            record["optimizer_ns"] = now_ns() - optimizer_start + (now_ns() - clip_start)
            record["wall_ns"] = now_ns() - step_start
            record["non_compute_ns_proxy"] = max(
                0,
                record["wall_ns"] - record["forward_cuda_event_ns"] - record["backward_cuda_event_ns"],
            )
            if phase == "timed":
                step_records.append(record)
                for key in phase_totals:
                    phase_totals[key] += int(record[key])
    total_ns = now_ns() - total_start
    summary = None
    finalize_ns = 0
    report_json_ns = 0
    summary_sha = None
    if telemetry:
        finalize_start = now_ns()
        summary = backend.get_summary("B")
        finalize_ns = now_ns() - finalize_start
        if backend_name != "buffered_full":
            report_start = now_ns()
            json.dumps(summary, sort_keys=True, separators=(",", ":"))
            report_json_ns = now_ns() - report_start
            summary_sha = json_sha256(summary)
        else:
            # The buffered backend leaves JSON serialization for the report
            # flush after all timed runs. The summary tensor extraction above
            # is already outside the optimizer wall-time window.
            summary_sha = None
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend_name if telemetry else "baseline_off",
        "telemetry": telemetry,
        "repetition": repetition,
        "config_sha256": config_sha,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": state_digest(model),
        "total_wall_ns": total_ns,
        "telemetry_finalize_ns": finalize_ns,
        "report_json_ns": report_json_ns,
        "summary_sha256": summary_sha,
        "phase_sums": phase_totals,
        "timed_records": step_records,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "numerical": {
            "all_finite_losses": all(math.isfinite(v) for r in step_records for v in r["loss_values"]),
        },
        "_summary": summary,
    }
    return result


def profiler_probe(
    *,
    config: Mapping[str, Any],
    root: Path,
    raw: torch.Tensor,
    batch: Any,
    anchor_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Profile one disposable forward/backward/optimizer workload."""
    hp = config["hyperparameters"]
    model = load_model(anchor_path, hp, device)
    optimizer = make_optimizer(model, hp, float(config["contract"]["learning_rate_candidates"][0]))
    BatchedGpuTelemetry.configure(int(hp["vocab_size"]))
    BatchedGpuTelemetry.reset_step_buffer()
    x_cpu, y_cpu = assemble_current(raw, batch)
    x_gpu = x_cpu.to(device)
    y_gpu = y_cpu.to(device)
    seq_idx = torch.arange(x_gpu.size(1), device=device).unsqueeze(0).expand(x_gpu.size(0), -1)
    sequence_ids = torch.arange(x_gpu.size(0), device=device)

    originals = []
    for index, layer in enumerate(model.layers):
        for attr, label in (("attn", f"attention_layer_{index}"), ("moe", f"moe_layer_{index}")):
            module = getattr(layer, attr)
            original = module.forward
            originals.append((module, original))

            def wrapped(*args: Any, _original=original, _label=label, **kwargs: Any) -> Any:
                with torch.profiler.record_function(_label):
                    return _original(*args, **kwargs)

            module.forward = wrapped  # type: ignore[method-assign]

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    backward_start_event = torch.cuda.Event(enable_timing=True)
    backward_end_event = torch.cuda.Event(enable_timing=True)
    optimizer_start_event = torch.cuda.Event(enable_timing=True)
    optimizer_end_event = torch.cuda.Event(enable_timing=True)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as profiler:
        optimizer.zero_grad()
        with torch.profiler.record_function("forward_total"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    x_gpu,
                    seq_idx,
                    current_branch="B",
                    current_sequence_ids=sequence_ids,
                    telemetry_class=BatchedGpuTelemetry,
                )
                ce_sum, _, valid_count, _ = loss_sums(logits, y_gpu)
                aux = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
                loss = (ce_sum + float(hp["branch_b_aux_coef"]) * aux * valid_count) / int(batch.nonpad_target_tokens)
        with torch.profiler.record_function("backward_total"):
            backward_start_event.record()
            loss.backward()
            backward_end_event.record()
            backward_end_event.synchronize()
        with torch.profiler.record_function("optimizer_total"):
            optimizer_start_event.record()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["max_norm_clip"]))
            optimizer.step()
            optimizer_end_event.record()
            optimizer_end_event.synchronize()
        torch.cuda.synchronize(device)

    for module, original in originals:
        module.forward = original  # type: ignore[method-assign]

    def device_us(event: Any, self_only: bool) -> float:
        names = (
            ("self_device_time_total", "self_cuda_time_total")
            if self_only else ("device_time_total", "cuda_time_total")
        )
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                return float(value)
        return 0.0

    attention = 0.0
    moe = 0.0
    backward = 0.0
    optimizer_time = 0.0
    routing_ops: Dict[str, float] = {}
    top_ops: List[Dict[str, Any]] = []
    for event in profiler.key_averages():
        key = str(event.key)
        total = device_us(event, False)
        self_time = device_us(event, True)
        top_ops.append({"op": key, "self_cuda_us": self_time, "cuda_us": total})
        if key.startswith("attention_layer_"):
            attention += total
        if key.startswith("moe_layer_"):
            moe += total
        if key == "backward_total":
            backward += total
        if key == "optimizer_total":
            optimizer_time += total
        if any(token in key.lower() for token in ("bincount", "index_add", "topk", "gather", "index_put", "scatter")):
            routing_ops[key] = total
    top_ops.sort(key=lambda item: item["self_cuda_us"], reverse=True)
    return {
        "attention_labeled_cuda_us": attention,
        "moe_labeled_cuda_us": moe,
        "backward_record_function_cuda_us": backward,
        "optimizer_record_function_cuda_us": optimizer_time,
        "backward_cuda_event_us": float(backward_start_event.elapsed_time(backward_end_event) * 1000.0),
        "optimizer_cuda_event_us": float(optimizer_start_event.elapsed_time(optimizer_end_event) * 1000.0),
        "routing_indexing_ops_cuda_us": sum(routing_ops.values()),
        "routing_indexing_ops": routing_ops,
        "top_cuda_ops": top_ops[:25],
        "profile_scope": "one disposable batch, forward + backward + optimizer",
    }


def public_run(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_summary"}


def main() -> int:
    parser = argparse.ArgumentParser(description="SSRE-NPA v5 telemetry-efficiency benchmark")
    parser.add_argument("--context-root", type=Path, default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--skip-profiler", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    root = args.context_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    hp = config["hyperparameters"]
    if hp["num_experts"] != 16 or hp["top_k"] != 2 or config["contract"]["architectural_branches"] != ["B"]:
        raise ValueError("frozen E16 Branch-B contract mismatch")
    anchor_path = root / "crucible_manifest" / config["anchors"]["42"]["relative_path"]
    batches, schedule_report, raw, source_report, binary_report = build_batches(
        config, root, 42, args.warmup_steps + args.timed_steps
    )
    index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_report = validate_prefix_inputs(raw, batches, index_tensors, min(32, len(batches)))
    device = torch.device("cuda")

    backend_names = ["baseline_off", "current_full", "gpu_full_batched", "minimal_scientific", "buffered_full"]
    internal: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for backend_name in backend_names:
            torch.manual_seed(12345)
            result = run_trial(
                backend_name=backend_name,
                telemetry=backend_name != "baseline_off",
                repetition=repetition,
                config=config,
                config_sha=config_sha,
                root=root,
                raw=raw,
                batches=batches,
                anchor_path=anchor_path,
                device=device,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
            )
            internal.append(result)
            print(json.dumps({
                "backend": backend_name,
                "repetition": repetition,
                "wall_s": round(result["total_wall_ns"] / 1e9, 3),
                "final_state": result["final_state_sha256"][:16],
                "finalize_ms": round(result["telemetry_finalize_ns"] / 1e6, 3),
            }, sort_keys=True), flush=True)
            torch.cuda.empty_cache()

    equivalence: List[Dict[str, Any]] = []
    telemetry_equivalence: Dict[str, Any] = {}
    for repetition in range(1, args.repetitions + 1):
        baseline = next(item for item in internal if item["backend"] == "baseline_off" and item["repetition"] == repetition)
        for candidate_name in backend_names[1:]:
            candidate = next(item for item in internal if item["backend"] == candidate_name and item["repetition"] == repetition)
            left_losses = [value for record in baseline["timed_records"] for value in record["loss_values"]]
            right_losses = [value for record in candidate["timed_records"] for value in record["loss_values"]]
            max_delta = max((abs(a - b) for a, b in zip(left_losses, right_losses)), default=0.0)
            equivalence.append({
                "repetition": repetition,
                "candidate": candidate_name,
                "final_state_sha_equal": baseline["final_state_sha256"] == candidate["final_state_sha256"],
                "max_loss_abs_delta": max_delta,
                "loss_tolerance": 1e-5,
                "losses_within_tolerance": max_delta <= 1e-5,
            })

    for repetition in range(1, args.repetitions + 1):
        current = next(item for item in internal if item["backend"] == "current_full" and item["repetition"] == repetition)
        for candidate_name in ("gpu_full_batched", "buffered_full", "minimal_scientific"):
            candidate = next(item for item in internal if item["backend"] == candidate_name and item["repetition"] == repetition)
            comparison = compare_telemetry(
                current["_summary"],
                candidate["_summary"],
                diagnostic=candidate_name != "minimal_scientific",
            )
            telemetry_equivalence[f"rep_{repetition}_{candidate_name}"] = comparison

    buffered_flushes = []
    for item in internal:
        if item["backend"] != "buffered_full":
            continue
        flush_start = now_ns()
        payload = json.dumps(item["_summary"], sort_keys=True, separators=(",", ":"))
        item["report_json_ns"] = now_ns() - flush_start
        item["summary_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        buffered_flushes.append({
            "repetition": item["repetition"],
            "deferred_json_ns": item["report_json_ns"],
            "summary_sha256": item["summary_sha256"],
        })

    profiler_report = None
    if not args.skip_profiler:
        profiler_report = profiler_probe(
            config=config,
            root=root,
            raw=raw,
            batch=batches[0],
            anchor_path=anchor_path,
            device=device,
        )

    public_runs = [public_run(item) for item in internal]
    summary = {
        "schema_version": "ssre-npa-v5-summary",
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "deterministic_control": "torch deterministic algorithms warn_only; frozen CUDA telemetry bincount has no deterministic implementation",
        "context_root": str(root),
        "config_sha256": config_sha,
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "source_report": source_report,
        "binary_report": binary_report,
        "schedule_report": schedule_report,
        "input_report": input_report,
        "parameters": {
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "repetitions": args.repetitions,
            "model_parameter_count": int(hp["expected_parameter_geometry"]),
            "context": int(hp["context_window_length"]),
            "num_experts": int(hp["num_experts"]),
            "top_k": int(hp["top_k"]),
        },
        "runs": public_runs,
        "equivalence": equivalence,
        "telemetry_equivalence": telemetry_equivalence,
        "buffered_report_flushes": buffered_flushes,
        "profiler": profiler_report,
        "integrity": {
            "all_runs_finite": all(item["numerical"]["all_finite_losses"] for item in internal),
            "all_input_prefixes_exact": input_report["exact_equivalence"],
            "all_model_state_equivalent": all(item["final_state_sha_equal"] for item in equivalence),
            "all_loss_tolerances_pass": all(item["losses_within_tolerance"] for item in equivalence),
            "all_full_telemetry_counters_pass": all(
                value["pass"] for key, value in telemetry_equivalence.items()
                if "minimal_scientific" not in key
            ),
            "all_primary_telemetry_counters_pass": all(
                value["exact_counter_fields_equal"] and value["float_fields_within_tolerance"]
                for value in telemetry_equivalence.values()
            ),
        },
        "limitations": [
            "This is a short local engineering benchmark, not a scientific training run.",
            "Reporting/JSON timing is outside the end-to-end step wall time, matching the frozen runner's checkpoint reporting boundary.",
            "Buffered reporting is measured as deferred batched extraction/serialization; actual filesystem fsync is not placed in the GPU critical path here.",
            "Profiler category labels cover forward attention and MoE modules; backward is reported as a measured total, while routing/indexing is summarized from profiled CUDA operators.",
        ],
    }
    out_path = output / "telemetry_efficiency_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "runs": len(public_runs),
        "all_finite": summary["integrity"]["all_runs_finite"],
        "model_equivalence": summary["integrity"]["all_model_state_equivalent"],
        "full_telemetry_equivalence": summary["integrity"]["all_full_telemetry_counters_pass"],
        "primary_telemetry_equivalence": summary["integrity"]["all_primary_telemetry_counters_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
