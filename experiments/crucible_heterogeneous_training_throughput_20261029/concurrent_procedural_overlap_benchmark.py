#!/usr/bin/env python3
"""Measure useful CPU procedural work overlapped with GPU Crucible training.

This is an engineering benchmark, not a scientific Crucible experiment.  The
GPU trains on the frozen context-128 workload while CPU workers construct and
independently verify deterministic synthetic procedures attached to future
training microbatches.  The sequential arm performs exactly the same CPU work
after each GPU step; the concurrent arm uses bounded SSRE-style prefetching.

The CPU procedure result is consumed as ordered metadata after the associated
GPU step.  It is deliberately not injected into model gradients: this isolates
useful-work overlap from a claim that the synthetic procedure task improves
neural learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import (  # noqa: E402
    Batch,
    LOSS_SCALE,
    assemble_current,
    batch_fingerprint,
    build_batches,
    bytes_for_tensors,
    cuda_event_pair,
    loss_sum,
    make_model,
    model_config,
    parameter_groups,
    state_digest,
    unscale_grads,
    write_json,
)


SCHEMA_VERSION = "crucible-concurrent-procedural-overlap-v1-20261029"
OPS = ("add_mod10", "sub_mod10", "mul_mod10", "xor_mod10", "max_mod10", "min_mod10")
DOMAIN = tuple((a, b) for a in range(10) for b in range(10))
TaskPayload = Tuple[int, int, int, int, Tuple[int, ...], Tuple[bytes, ...], int]
WORKLOAD_MODE = "synthetic"
ACTUAL_RUNNER: Any = None


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def set_deterministic_cuda() -> None:
    """Make baseline/concurrent state comparisons meaningful on CUDA."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reference_eval(name: str, a: int, b: int) -> int:
    if name == "add_mod10":
        return (a + b) % 10
    if name == "sub_mod10":
        return (a - b) % 10
    if name == "mul_mod10":
        return (a * b) % 10
    if name == "xor_mod10":
        return (a ^ b) % 10
    if name == "max_mod10":
        return max(a, b) % 10
    if name == "min_mod10":
        return min(a, b) % 10
    raise ValueError(f"unknown operation: {name}")


def make_ast(name: str, candidate_slot: int) -> Tuple[Any, ...]:
    """Create one correct candidate and deterministic distractor candidates."""
    if candidate_slot == 0:
        return ("mod10", ("op", name, ("arg", 0), ("arg", 1)))
    distractor = OPS[candidate_slot % len(OPS)]
    if distractor == name:
        distractor = OPS[(candidate_slot + 1) % len(OPS)]
    if candidate_slot % 2:
        return ("mod10", ("op", distractor, ("arg", 1), ("arg", 0)))
    return ("mod10", ("op", distractor, ("arg", 0), ("arg", 1)))


def execute_ast(ast: Tuple[Any, ...], a: int, b: int) -> int:
    if ast[0] != "mod10":
        raise ValueError("invalid root")
    node = ast[1]
    if node[0] != "op":
        raise ValueError("invalid operation node")
    left, right = node[2], node[3]
    if left[0] != "arg" or right[0] != "arg":
        raise ValueError("invalid typed hole")
    values = (a, b)
    left_value, right_value = values[int(left[1])], values[int(right[1])]
    name = str(node[1])
    if name == "add_mod10":
        return (left_value + right_value) % 10
    if name == "sub_mod10":
        return (left_value - right_value) % 10
    if name == "mul_mod10":
        return (left_value * right_value) % 10
    if name == "xor_mod10":
        return (left_value ^ right_value) % 10
    if name == "max_mod10":
        return max(left_value, right_value) % 10
    if name == "min_mod10":
        return min(left_value, right_value) % 10
    raise ValueError(f"unknown AST operation: {name}")


def ast_digest(ast: Tuple[Any, ...]) -> str:
    return sha256_bytes(json.dumps(ast, separators=(",", ":")).encode("utf-8"))


def task_manifest_digest(payloads: Sequence[TaskPayload]) -> str:
    digest = hashlib.sha256()
    for task_id, step, microbatch, batch_index, indices, records, rounds in payloads:
        digest.update(f"{task_id}|{step}|{microbatch}|{batch_index}|{rounds}|".encode("ascii"))
        digest.update(",".join(str(item) for item in indices).encode("ascii"))
        for record in records:
            digest.update(record)
    return digest.hexdigest()


def actual_task_manifest_digest(payloads: Sequence[Tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    for task_id, step, microbatch, batch_index, source_task_id, demonstrations, verification_examples in payloads:
        digest.update(f"{task_id}|{step}|{microbatch}|{batch_index}|".encode("ascii"))
        digest.update(str(source_task_id).encode("utf-8"))
        digest.update(json.dumps(demonstrations, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(json.dumps(verification_examples, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def make_task_payloads(
    batches: Sequence[Batch],
    records: Sequence[bytes],
    rounds: int,
) -> List[TaskPayload]:
    payloads: List[TaskPayload] = []
    for task_id, batch in enumerate(batches):
        payloads.append(
            (
                task_id,
                int(batch.step),
                int(batch.microbatch_index),
                task_id,
                tuple(int(index) for index in batch.indices),
                tuple(records[index] for index in batch.indices),
                int(rounds),
            )
        )
    return payloads


def make_actual_task_payloads(
    batches: Sequence[Batch],
    procedure_tasks: Sequence[Mapping[str, Any]],
) -> List[Tuple[Any, ...]]:
    """Attach real Crucible proposal/verifier tasks to future microbatches."""
    if not procedure_tasks:
        raise ValueError("actual workload requires procedure tasks")
    payloads: List[Tuple[Any, ...]] = []
    for task_id, batch in enumerate(batches):
        task = procedure_tasks[task_id % len(procedure_tasks)]
        payloads.append(
            (
                task_id,
                int(batch.step),
                int(batch.microbatch_index),
                task_id,
                str(task["task_id"]),
                tuple(dict(example) for example in task.get("demonstrations", [])),
                tuple(dict(example) for example in task.get("verification_examples", [])),
            )
        )
    return payloads


def generate_and_verify_actual(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    """Run Crucible's actual grammar candidate and independent verifier path."""
    if ACTUAL_RUNNER is None:
        raise RuntimeError("actual Crucible workload is not configured")
    task_id, step, microbatch, batch_index, source_task_id, demonstrations, verification_examples = payload
    wall_start = now_ns()
    cpu_start = time.process_time_ns()
    proposal_task_id, candidates = ACTUAL_RUNNER._crucible_cpu_candidate_job(
        (str(source_task_id), tuple(demonstrations))
    )
    if proposal_task_id != str(source_task_id):
        raise RuntimeError("Crucible candidate worker returned an unexpected task ID")
    verifier = ACTUAL_RUNNER.IndependentVerifier()
    traces: List[Dict[str, Any]] = []
    accepted: Optional[Mapping[str, Any]] = None
    verifier_calls = 0
    verifier_examples = 0
    for slot, candidate in enumerate(candidates[:4]):
        ok, checks = verifier.verify(candidate["ast"], demonstrations)
        verifier_calls += 1
        verifier_examples += checks
        traces.append({
            "slot": slot,
            "accepted_on_demonstrations": ok,
            "checks": checks,
            "canonical_ast_hash": candidate["canonical_ast_hash"],
            "source": candidate.get("source"),
        })
        if ok and accepted is None:
            accepted = candidate
    independently_verified = False
    independent_checks = 0
    if accepted is not None and verification_examples:
        independently_verified, independent_checks = verifier.verify(
            accepted["ast"], verification_examples
        )
        verifier_calls += 1
        verifier_examples += independent_checks
    stable = {
        "task_id": str(source_task_id),
        "candidate_hashes": [str(candidate["canonical_ast_hash"]) for candidate in candidates[:4]],
        "accepted_hash": None if accepted is None else str(accepted["canonical_ast_hash"]),
        "independently_verified": independently_verified,
        "verification_examples": independent_checks,
    }
    return {
        "task_id": int(task_id),
        "step": int(step),
        "microbatch_index": int(microbatch),
        "batch_index": int(batch_index),
        "source_task_id": str(source_task_id),
        "candidate_count": len(candidates[:4]),
        "accepted_slots": [int(trace["slot"]) for trace in traces if trace["accepted_on_demonstrations"]],
        "verified_procedures": int(independently_verified),
        "verifier_calls": verifier_calls,
        "verifier_examples": verifier_examples,
        "independent_verification_checks": independent_checks,
        "candidate_hashes": stable["candidate_hashes"],
        "accepted_hash": stable["accepted_hash"],
        "result_digest": sha256_bytes(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "worker_pid": os.getpid(),
        "worker_cpu_ns": time.process_time_ns() - cpu_start,
        "task_wall_ns": now_ns() - wall_start,
    }


def generate_and_verify_procedure(payload: Any) -> Dict[str, Any]:
    """Generate four candidates and independently verify them exhaustively."""
    if WORKLOAD_MODE == "actual":
        return generate_and_verify_actual(payload)
    (
        task_id,
        step,
        microbatch,
        batch_index,
        indices,
        records,
        rounds,
    ) = payload
    wall_start = now_ns()
    cpu_start = time.process_time_ns()
    source_digest = hashlib.sha256()
    source_bytes = 0
    for index, record in zip(indices, records):
        if not record:
            raise ValueError("empty source record")
        source_digest.update(str(index).encode("ascii"))
        source_digest.update(record)
        source_bytes += len(record)

    operator = OPS[(task_id + step + microbatch) % len(OPS)]
    correct_slot = 0
    candidates = [make_ast(operator, slot) for slot in range(4)]
    candidate_digests = [ast_digest(candidate) for candidate in candidates]
    expected = [reference_eval(operator, a, b) for a, b in DOMAIN]
    accepted: List[int] = []
    for slot, candidate in enumerate(candidates):
        valid = True
        for _ in range(rounds):
            for pair_index, (a, b) in enumerate(DOMAIN):
                if execute_ast(candidate, a, b) != expected[pair_index]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            accepted.append(slot)

    if accepted != [correct_slot]:
        raise ValueError(f"independent verification ambiguity at task {task_id}: {accepted}")
    stable = f"{task_id}|{step}|{microbatch}|{operator}|{correct_slot}|".encode("ascii")
    stable += source_digest.digest() + "|".join(candidate_digests).encode("ascii")
    return {
        "task_id": task_id,
        "step": step,
        "microbatch_index": microbatch,
        "batch_index": batch_index,
        "candidate_count": len(candidates),
        "accepted_slots": accepted,
        "verified_procedures": 1,
        "operator": operator,
        "source_bytes": source_bytes,
        "source_record_count": len(records),
        "ast_digest": candidate_digests[correct_slot],
        "result_digest": sha256_bytes(stable),
        "worker_pid": os.getpid(),
        "worker_cpu_ns": time.process_time_ns() - cpu_start,
        "task_wall_ns": now_ns() - wall_start,
    }


class SSREProcedurePrefetcher:
    """Bounded ordered process reservoir for independent procedure work."""

    def __init__(self, payloads: Sequence[TaskPayload], workers: int, window: int):
        self.payloads = payloads
        self.workers = max(1, int(workers))
        self.window = max(1, int(window))
        pool_kwargs: Dict[str, Any] = {"max_workers": self.workers}
        if "fork" in mp.get_all_start_methods():
            pool_kwargs["mp_context"] = mp.get_context("fork")
        self.pool = ProcessPoolExecutor(**pool_kwargs)
        self.pending: Dict[int, Tuple[Future[Dict[str, Any]], int]] = {}
        self.next_submit = 0
        self.next_commit = 0
        self.max_occupancy = 0
        self.occupancy_samples: List[int] = []

    def _fill(self) -> None:
        while self.next_submit < len(self.payloads) and len(self.pending) < self.window:
            submitted = now_ns()
            future = self.pool.submit(generate_and_verify_procedure, self.payloads[self.next_submit])
            self.pending[self.next_submit] = (future, submitted)
            self.next_submit += 1
            self.max_occupancy = max(self.max_occupancy, len(self.pending))
            self.occupancy_samples.append(len(self.pending))

    def prime(self) -> None:
        """Submit the initial bounded window before GPU work begins."""
        self._fill()

    def get(self, task_id: int) -> Dict[str, Any]:
        if task_id != self.next_commit:
            raise RuntimeError(f"ordered commit violation: expected {self.next_commit}, got {task_id}")
        self._fill()
        future, submitted = self.pending.pop(task_id)
        wait_start = now_ns()
        result = future.result()
        result["queue_age_ns"] = wait_start - submitted
        result["blocking_wait_ns"] = now_ns() - wait_start
        self.next_commit += 1
        self._fill()
        self.occupancy_samples.append(len(self.pending))
        return result

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)

    def metrics(self) -> Dict[str, Any]:
        return {
            "declared_workers": self.workers,
            "max_queue_occupancy": self.max_occupancy,
            "mean_queue_occupancy": (
                sum(self.occupancy_samples) / len(self.occupancy_samples)
                if self.occupancy_samples
                else 0.0
            ),
        }


def prepare_batch(raw: torch.Tensor, batch: Batch, pin_memory: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    x_cpu, y_cpu, _, _ = assemble_current(raw, batch)
    if pin_memory:
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
    return x_cpu, y_cpu


def train_gpu_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    raw: torch.Tensor,
    group: Sequence[Tuple[int, Batch]],
    device: torch.device,
    total_valid: int,
    pin_memory: bool,
) -> Dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    record: Dict[str, Any] = {
        "valid_targets": 0,
        "timed_tokens": 0,
        "h2d_bytes": 0,
        "h2d_ns": 0,
        "forward_cuda_ns": 0,
        "backward_cuda_ns": 0,
        "gpu_idle_ns_proxy": 0,
        "batch_prepare_ns": 0,
        "wall_ns": 0,
        "finite_gradients": True,
    }
    step_start = now_ns()
    for _, batch in group:
        prep_start = now_ns()
        x_cpu, y_cpu = prepare_batch(raw, batch, pin_memory)
        record["batch_prepare_ns"] += now_ns() - prep_start
        record["h2d_bytes"] += bytes_for_tensors(x_cpu, y_cpu)
        h2d_start = now_ns()
        x_gpu = x_cpu.to(device, non_blocking=pin_memory)
        y_gpu = y_cpu.to(device, non_blocking=pin_memory)
        torch.cuda.synchronize(device)
        record["h2d_ns"] += now_ns() - h2d_start
        f_start, f_end = cuda_event_pair(device)
        forward_start = now_ns()
        f_start.record()
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = model(x_gpu)
            ce_sum, _ = loss_sum(logits, y_gpu)
            loss = ce_sum / float(total_valid)
        f_end.record()
        f_end.synchronize()
        record["forward_cuda_ns"] += int(f_start.elapsed_time(f_end) * 1_000_000.0)
        backward_start = now_ns()
        b_start, b_end = cuda_event_pair(device)
        b_start.record()
        (loss * LOSS_SCALE).backward()
        b_end.record()
        b_end.synchronize()
        record["backward_cuda_ns"] += int(b_start.elapsed_time(b_end) * 1_000_000.0)
        record["valid_targets"] += int(batch.nonpad_target_tokens)
        record["timed_tokens"] += int(batch.nonpad_target_tokens)
        del logits, loss, x_gpu, y_gpu

    record["finite_gradients"] = unscale_grads(model, LOSS_SCALE)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_start = now_ns()
    torch.cuda.synchronize(device)
    optimizer.step()
    torch.cuda.synchronize(device)
    record["optimizer_ns"] = now_ns() - optimizer_start
    record["wall_ns"] = now_ns() - step_start
    record["gpu_idle_ns_proxy"] = max(
        0,
        record["wall_ns"] - record["forward_cuda_ns"] - record["backward_cuda_ns"],
    )
    return record


def compare_task_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item["task_id"]))
    return {
        "task_count": len(ordered),
        "verified_procedures": sum(int(item["verified_procedures"]) for item in ordered),
        "candidate_count": sum(int(item["candidate_count"]) for item in ordered),
        "result_digest": sha256_bytes("|".join(str(item["result_digest"]) for item in ordered).encode("ascii")),
        "worker_pids": sorted({int(item["worker_pid"]) for item in ordered}),
        "worker_cpu_ns": sum(int(item["worker_cpu_ns"]) for item in ordered),
        "task_wall_ns": sum(int(item["task_wall_ns"]) for item in ordered),
        "queue_age_ns": sum(int(item.get("queue_age_ns", 0)) for item in ordered),
        "blocking_wait_ns": sum(int(item.get("blocking_wait_ns", 0)) for item in ordered),
    }


def run_condition(
    condition: str,
    cfg: Mapping[str, int],
    batches: Sequence[Batch],
    raw: torch.Tensor,
    payloads: Sequence[TaskPayload],
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
    workers: int,
    window: int,
    pin_memory: bool,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    model = make_model(cfg, device, 42)
    initial_state = state_digest(model)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    batch_by_step: Dict[int, List[Tuple[int, Batch]]] = {}
    for index, batch in enumerate(batches):
        batch_by_step.setdefault(int(batch.step), []).append((index, batch))
    max_step = warmup_steps + timed_steps
    if max(batch_by_step) < max_step:
        raise ValueError(f"schedule prefix has no step {max_step}")

    prefetcher: Optional[SSREProcedurePrefetcher] = None
    if condition == "concurrent_ssre":
        prefetcher = SSREProcedurePrefetcher(payloads, workers, window)
    task_results: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    total_start = now_ns()
    parent_cpu_start = time.process_time_ns()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if prefetcher is not None:
        prefetcher.prime()
    task_cursor = 0
    try:
        for step in range(1, max_step + 1):
            group = batch_by_step[step]
            total_valid = sum(int(item.nonpad_target_tokens) for _, item in group)
            gpu_record = train_gpu_step(model, optimizer, raw, group, device, total_valid, pin_memory)
            gpu_record["step"] = step
            gpu_record["phase"] = "warmup" if step <= warmup_steps else "timed"
            step_records.append(gpu_record)
            current_task_ids = list(range(task_cursor, task_cursor + len(group)))
            if condition == "sequential_cpu":
                for task_id in current_task_ids:
                    task_results.append(generate_and_verify_procedure(payloads[task_id]))
            elif condition == "concurrent_ssre":
                assert prefetcher is not None
                for task_id in current_task_ids:
                    task_results.append(prefetcher.get(task_id))
            task_cursor += len(group)
    finally:
        if prefetcher is not None:
            prefetcher.close()
    total_wall_ns = now_ns() - total_start
    parent_cpu_ns = time.process_time_ns() - parent_cpu_start
    timed_steps_out = [item for item in step_records if item["phase"] == "timed"]
    timed_tasks = [item for item in task_results if int(item["step"]) > warmup_steps]
    timed_tokens = sum(int(item["timed_tokens"]) for item in timed_steps_out)
    task_summary = compare_task_results(timed_tasks)
    task_summary["all_task_count"] = len(task_results)
    task_summary["all_result_digest"] = compare_task_results(task_results)["result_digest"]
    task_summary["verified_procedures_per_end_to_end_second"] = task_summary["verified_procedures"] / max(1e-9, total_wall_ns / 1e9)
    worker_cpu_ns = int(task_summary["worker_cpu_ns"])
    actual_workers = len(task_summary["worker_pids"])
    if condition == "sequential_cpu":
        actual_workers = 1
    task_summary["actual_worker_count"] = actual_workers
    task_summary["worker_cpu_utilization_pct"] = (
        100.0 * worker_cpu_ns / max(1, total_wall_ns * max(1, actual_workers))
        if condition == "concurrent_ssre"
        else 100.0 * worker_cpu_ns / max(1, total_wall_ns)
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "device": "cuda",
        "deterministic_cuda": True,
        "workers": workers if condition == "concurrent_ssre" else 0,
        "window": window if condition == "concurrent_ssre" else 0,
        "pin_memory": pin_memory,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_state,
        "final_state_sha256": state_digest(model),
        "total_wall_ns": total_wall_ns,
        "parent_cpu_ns": parent_cpu_ns,
        "timed_tokens": timed_tokens,
        "neural_tokens_per_end_to_end_second": timed_tokens / max(1e-9, total_wall_ns / 1e9),
        "neural_tokens_per_gpu_step_second": timed_tokens / max(1e-9, sum(int(item["wall_ns"]) for item in timed_steps_out) / 1e9),
        "verified_procedures_per_end_to_end_second": task_summary["verified_procedures_per_end_to_end_second"],
        "step_records": timed_steps_out,
        "task_summary": task_summary,
        "gpu_metrics": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "forward_cuda_ns": sum(int(item["forward_cuda_ns"]) for item in timed_steps_out),
            "backward_cuda_ns": sum(int(item["backward_cuda_ns"]) for item in timed_steps_out),
            "gpu_idle_ns_proxy": sum(int(item["gpu_idle_ns_proxy"]) for item in timed_steps_out),
            "h2d_ns": sum(int(item["h2d_ns"]) for item in timed_steps_out),
            "h2d_bytes": sum(int(item["h2d_bytes"]) for item in timed_steps_out),
        },
    }
    if prefetcher is not None:
        result["prefetch_metrics"] = prefetcher.metrics()
    else:
        result["prefetch_metrics"] = {"declared_workers": 0, "max_queue_occupancy": 0, "mean_queue_occupancy": 0.0}
    return result


def render_report(report: Mapping[str, Any]) -> str:
    rows = []
    workload_name = str(report["workload_metadata"]["name"])
    workload_note = (
        "Crucible's actual frozen-grammar candidate generation and independent verifier path."
        if "actual" in workload_name
        else "A deterministic synthetic procedure generator and exhaustive verifier proxy."
    )
    for result in report["summary"]:
        rows.append(
            "| {condition} | {wall:.3f} | {tokens:.1f} | {procedures:.1f} | {workers} | {util:.1f}% | {queue} | {idle:.3f} |".format(
                condition=result["condition"],
                wall=result["mean_total_wall_ns"] / 1e9,
                tokens=result["mean_neural_tokens_per_end_to_end_second"],
                procedures=result["mean_verified_procedures_per_end_to_end_second"],
                workers=result["mean_actual_worker_count"],
                util=result["mean_worker_cpu_utilization_pct"],
                queue=result["max_queue_occupancy"],
                idle=result["mean_gpu_idle_ns_proxy"] / 1e9,
            )
        )
    seq = next(item for item in report["summary"] if item["condition"] == "sequential_cpu")
    con = next(item for item in report["summary"] if item["condition"] == "concurrent_ssre")
    speedup = seq["mean_total_wall_ns"] / max(1, con["mean_total_wall_ns"])
    return "\n".join(
        [
            "# Concurrent CPU procedural-work / GPU training benchmark",
            "",
            f"Classification: `{report['classification']}`",
            "",
            f"The CPU work is {workload_note} It is attached to future training microbatches and consumed in order; it does not affect model gradients.",
            "",
            "## Results",
            "",
            "| Condition | End-to-end wall (s) | Neural tokens/s | Verified procedures/s | Workers | CPU worker utilization | Max queue | GPU idle proxy (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Sequential/concurrent wall-time speedup: **{speedup:.3f}x**.",
            "",
            "## Controls",
            "",
            f"- Same task manifest: `{report['task_manifest_sha256']}`",
            f"- All-batch input equivalence: `{report['input_equivalence']['exact_equivalence']}`",
            f"- Deterministic CUDA algorithms: `{report['deterministic_cuda']}`",
            f"- Sequential/concurrent task-result digest match: `{report['task_result_digest_match']}`",
            f"- GPU-only versus concurrent final-state match: `{report['gpu_only_vs_concurrent_final_state_match']}`",
            f"- Repetitions per condition: `{report['benchmark_config']['repetitions']}`",
            "",
            "## Interpretation boundary",
            "",
            "This measures useful-work overlap, not neural learning quality. It is an engineering workload measurement, not a curriculum or scientific training result.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent CPU procedural work / GPU training benchmark")
    parser.add_argument("--context-root", type=Path, default=Path(os.environ.get("CRUCIBLE_CONTEXT_ROOT", "context_128")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("tiny", "small"), default="small")
    parser.add_argument("--workload", choices=("synthetic", "actual"), default="synthetic")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--verification-rounds", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory", default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark with host GPU access")
    set_deterministic_cuda()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    max_steps = args.warmup_steps + args.timed_steps
    _, config_sha, records, raw, batches, data_report = build_batches(args.context_root.resolve(), max_steps)
    index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_reference: List[str] = []
    input_indexed: List[str] = []
    for batch, index_tensor in zip(batches, index_tensors):
        current_x, current_y, _, _ = assemble_current(raw, batch)
        indexed = raw.index_select(0, index_tensor)
        indexed_x, indexed_y = indexed[:, 0].contiguous(), indexed[:, 1].contiguous()
        input_reference.append(batch_fingerprint(current_x, current_y)["fingerprint_sha256"])
        input_indexed.append(batch_fingerprint(indexed_x, indexed_y)["fingerprint_sha256"])
    input_equivalence = {
        "checked_batches": len(batches),
        "exact_equivalence": input_reference == input_indexed,
        "reference_sha256": sha256_bytes("|".join(input_reference).encode("ascii")),
        "indexed_sha256": sha256_bytes("|".join(input_indexed).encode("ascii")),
    }
    if not input_equivalence["exact_equivalence"]:
        raise RuntimeError("direct/indexed input mismatch")
    workload_metadata: Dict[str, Any] = {
        "name": "synthetic_ast_proxy",
        "target_answers_accessed": False,
    }
    global WORKLOAD_MODE, ACTUAL_RUNNER
    WORKLOAD_MODE = str(args.workload)
    if args.workload == "actual":
        from tools import crucible_cumulative_adaptation_scientific_runner_20261028 as actual_runner

        ACTUAL_RUNNER = actual_runner
        frozen_pool = actual_runner.load_frozen_pool()
        preflight_root = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"
        source_rows = actual_runner.load_jsonl(preflight_root / "SOURCE_MANIFEST.jsonl")
        validation_rows = actual_runner.load_jsonl(preflight_root / "OPERATOR_VALIDATION_MANIFEST.jsonl")
        source_tasks = actual_runner.make_source_tasks(source_rows)
        validation_tasks = actual_runner.make_validation_tasks(validation_rows)
        actual_runner_instance = actual_runner.FourArmRunner(frozen_pool, ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028", scientific=False)
        actual_state = actual_runner_instance._prepare_arm(
            "A3_VERIFIED_ADAPTIVE_LIBRARY",
            source_tasks,
            validation_tasks,
        )
        actual_runner._init_crucible_cpu_worker(
            actual_runner_instance.pool,
            actual_runner_instance.pool_truth_tables,
            tuple(actual_state["library"].promoted),
            "A3_VERIFIED_ADAPTIVE_LIBRARY",
        )
        source_work_tasks: List[Dict[str, Any]] = []
        source_pairs = [(9, 9), (8, 7), (6, 5), (4, 2)]
        for task in source_tasks:
            source_copy = dict(task)
            source_copy["verification_examples"] = actual_runner.examples_for_ast(task["target_ast"], source_pairs)
            source_work_tasks.append(source_copy)
        actual_work_tasks = source_work_tasks + [dict(task) for task in validation_tasks]
        payloads = make_actual_task_payloads(batches, actual_work_tasks)
        payload_digest = actual_task_manifest_digest(payloads)
        workload_metadata = {
            "name": "crucible_actual_candidate_generation_and_independent_verification",
            "source_task_count": len(source_tasks),
            "operator_validation_task_count": len(validation_tasks),
            "workload_task_pool_count": len(actual_work_tasks),
            "frozen_pool_count": len(frozen_pool),
            "operator_activation_status": actual_state["library"].activation_status,
            "promoted_operator_count": len(actual_state["library"].promoted),
            "candidate_budget": 4,
            "target_answers_accessed": False,
            "sealed_target_answers_accessed": False,
            "source_and_validation_ast_metadata_used_for_library": True,
            "worker_input_fields": ["task_id", "demonstrations", "verification_examples"],
        }
    else:
        payloads = make_task_payloads(batches, records, args.verification_rounds)
        payload_digest = task_manifest_digest(payloads)
    cfg = model_config(args.model_size)
    results = []
    for repetition in range(1, args.repetitions + 1):
        for condition in ("gpu_only", "sequential_cpu", "concurrent_ssre"):
            result = run_condition(
                condition,
                cfg,
                batches,
                raw,
                payloads,
                device,
                args.warmup_steps,
                args.timed_steps,
                args.workers,
                args.window,
                args.pin_memory,
                args.learning_rate,
                args.weight_decay,
            )
            result["repetition"] = repetition
            results.append(result)

    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        by_condition.setdefault(str(result["condition"]), []).append(result)

    def mean(condition: str, key: str) -> float:
        values = [float(item[key]) for item in by_condition[condition]]
        return sum(values) / len(values)

    summary = []
    for condition in ("gpu_only", "sequential_cpu", "concurrent_ssre"):
        rows = by_condition[condition]
        summary.append(
            {
                "condition": condition,
                "repetitions": len(rows),
                "mean_total_wall_ns": mean(condition, "total_wall_ns"),
                "mean_neural_tokens_per_end_to_end_second": mean(condition, "neural_tokens_per_end_to_end_second"),
                "mean_neural_tokens_per_gpu_step_second": mean(condition, "neural_tokens_per_gpu_step_second"),
                "mean_verified_procedures_per_end_to_end_second": mean(condition, "verified_procedures_per_end_to_end_second"),
                "mean_actual_worker_count": sum(int(item["task_summary"]["actual_worker_count"]) for item in rows) / len(rows),
                "mean_worker_cpu_utilization_pct": sum(float(item["task_summary"]["worker_cpu_utilization_pct"]) for item in rows) / len(rows),
                "max_queue_occupancy": max(int(item["prefetch_metrics"]["max_queue_occupancy"]) for item in rows),
                "mean_gpu_idle_ns_proxy": sum(int(item["gpu_metrics"]["gpu_idle_ns_proxy"]) for item in rows) / len(rows),
                "verified_procedures": sum(int(item["task_summary"]["verified_procedures"]) for item in rows) // len(rows),
                "timed_task_count": sum(int(item["task_summary"]["task_count"]) for item in rows) // len(rows),
            }
        )
    paired_sequential = {int(item["repetition"]): item for item in by_condition["sequential_cpu"]}
    paired_concurrent = {int(item["repetition"]): item for item in by_condition["concurrent_ssre"]}
    paired_gpu = {int(item["repetition"]): item for item in by_condition["gpu_only"]}
    task_result_digest_match = all(
        paired_sequential[key]["task_summary"]["all_result_digest"] == paired_concurrent[key]["task_summary"]["all_result_digest"]
        for key in paired_sequential
    )
    gpu_only_vs_concurrent_final_state_match = all(
        paired_gpu[key]["final_state_sha256"] == paired_concurrent[key]["final_state_sha256"]
        for key in paired_gpu
    )
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "exploratory_gpu_overlap_measured",
        "scientific_status": "not_a_scientific_training_run",
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "parameter_count": sum(p.numel() for p in make_model(cfg, device, 42).parameters()), "config": cfg},
        "benchmark_config": {
            "conditions": ["gpu_only", "sequential_cpu", "concurrent_ssre"],
            "workload": args.workload,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "workers": args.workers,
            "window": args.window,
            "verification_rounds": args.verification_rounds,
            "repetitions": args.repetitions,
            "pin_memory": args.pin_memory,
            "deterministic_cuda": True,
        },
        "data_manifest_sha256": config_sha,
        "data_report": {key: {k: v for k, v in value.items() if k != "path"} for key, value in data_report.items()},
        "task_manifest_sha256": payload_digest,
        "task_definition": (
            "Crucible candidate generation from the frozen representative grammar pool followed by independent demonstration and verification-example checks"
            if args.workload == "actual"
            else "four grammar-constrained candidates; one independently verified over all 100 digit pairs; source records hashed per future training microbatch"
        ),
        "workload_metadata": workload_metadata,
        "input_equivalence": input_equivalence,
        "deterministic_cuda": True,
        "results": results,
        "summary": summary,
        "task_result_digest_match": task_result_digest_match,
        "gpu_only_vs_concurrent_final_state_match": gpu_only_vs_concurrent_final_state_match,
        "limitations": [
            "The procedure workload is a deterministic CPU proxy, not a scientific curriculum or learning result.",
            "The CPU result is consumed as ordered metadata and does not affect neural gradients.",
            "The model is a small representative Crucible-family model, not the future 500M model.",
            "End-to-end useful-work comparisons are valid only for this task mix and verification-round budget.",
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "OVERLAP_REPORT.json", report)
    (output / "OVERLAP_REPORT.md").write_text(render_report(report), encoding="utf-8")
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "HASHES.json":
            hashes[path.name] = sha256_bytes(path.read_bytes())
    write_json(output / "HASHES.json", {"schema_version": SCHEMA_VERSION, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
