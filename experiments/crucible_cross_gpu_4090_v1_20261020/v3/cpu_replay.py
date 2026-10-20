#!/usr/bin/env python3
"""Measured CPU-only replay of Crucible's frozen batch-delivery path.

The replay uses the actual context-128 binary dataset, packed source records
and both frozen seed schedules. It does not import the Crucible model, call
CUDA, execute a forward/backward pass, alter a manifest, or start training.

Arms:
  current_python_stack  - mirrors ImmutableBinaryDataset slicing/cloning and
                          torch.stack in the current runner.
  indexed_precomputed   - resident [example, x/y, position] tensor with
                          indexed batch extraction.
  cpu_reservoir         - bounded ordered future reservoir using persistent
                          CPU worker threads over the indexed representation.

Every arm emits a digest for x, y, PAD/non-PAD masks, raw-byte masks and the
ordered index list, so throughput cannot be improved by silently changing
scientific inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3
SCHEMA_VERSION = "ssre-npa-measured-batch-replay-v3"
SOURCE_KIND = "measured_cpu_batch_replay"


class NullTrace:
    """Sink used for executed-but-discarded warmup batches."""

    def write(self, _: str) -> None:
        return None


@dataclass(frozen=True)
class Batch:
    seed: int
    step: int
    microbatch_index: int
    indices: Tuple[int, ...]
    source_bytes: int
    nonpad_target_tokens: int
    raw_byte_target_tokens: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_ns() -> int:
    return time.perf_counter_ns()


def current_rss_bytes() -> int:
    # Linux reports KiB for ru_maxrss; this package runs on the Pop!_OS host.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def tensor_digest(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().contiguous().numpy().tobytes())


def batch_fingerprint(x: torch.Tensor, y: torch.Tensor) -> Dict[str, str]:
    valid = y != PAD_ID
    raw = y >= RAW_BYTE_START
    return {
        "x_sha256": tensor_digest(x),
        "y_sha256": tensor_digest(y),
        "valid_target_mask_sha256": tensor_digest(valid),
        "raw_target_mask_sha256": tensor_digest(raw),
        "fingerprint_sha256": sha256_bytes(
            bytes.fromhex(tensor_digest(x))
            + bytes.fromhex(tensor_digest(y))
            + bytes.fromhex(tensor_digest(valid))
            + bytes.fromhex(tensor_digest(raw))
        ),
        "valid_target_count": str(int(valid.sum().item())),
        "raw_target_count": str(int(raw.sum().item())),
    }


def index_digest(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return sha256_bytes(payload)


def load_manifest(root: Path) -> Tuple[Dict[str, Any], str]:
    config_path = root / "crucible_manifest" / "experiment_config.json"
    config_sha = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, config_sha


def load_source_records(root: Path, split: Mapping[str, Any]) -> Tuple[List[bytes], Dict[str, Any]]:
    path = root / split["source_jsonl_relative_path"]
    expected_sha = str(split["source_jsonl_sha256"])
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"source SHA mismatch: {path}")
    started = now_ns()
    records: List[bytes] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text_bytes = str(record["text"]).encode("utf-8")
            if int(record["example_index"]) != len(records):
                raise ValueError(f"example index mismatch at {path}:{line_number}")
            if int(record["byte_length"]) != len(text_bytes):
                raise ValueError(f"byte length mismatch at {path}:{line_number}")
            if record["text_utf8_sha256"] != sha256_bytes(text_bytes):
                raise ValueError(f"text hash mismatch at {path}:{line_number}")
            if text_bytes in seen:
                raise ValueError(f"duplicate text at {path}:{line_number}")
            seen.add(text_bytes)
            records.append(text_bytes)
    if len(records) != int(split["total_constructed_examples"]):
        raise ValueError(f"source count mismatch: {path}")
    return records, {
        "path": str(path),
        "sha256": actual_sha,
        "count": len(records),
        "source_bytes": sum(len(item) for item in records),
        "validation_ns": now_ns() - started,
    }


def load_binary(root: Path, split: Mapping[str, Any], count: int, context: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    path = root / split["relative_path"]
    expected_sha = str(split["binary_payload_sha256"])
    started = now_ns()
    raw = path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_sha:
        raise ValueError(f"binary SHA mismatch: {path}")
    values = torch.frombuffer(bytearray(raw), dtype=torch.int64)
    expected_values = count * 2 * context
    if values.numel() != expected_values:
        raise ValueError(f"binary geometry mismatch: {path}")
    view = values.view(count, 2, context)
    return view, {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": len(raw),
        "samples": count,
        "load_ns": now_ns() - started,
    }


def verify_source_binary(records: Sequence[bytes], binary: torch.Tensor, context: int) -> Dict[str, Any]:
    started = now_ns()
    if len(records) != binary.size(0):
        raise ValueError("source/binary count mismatch")
    for example_index, text_bytes in enumerate(records):
        if len(text_bytes) > context - 1:
            raise ValueError(f"source exceeds BOS context capacity at {example_index}")
        byte_tokens = [RAW_BYTE_START + value for value in text_bytes]
        expected_x = [BOS_ID] + byte_tokens + [PAD_ID] * (context - 1 - len(byte_tokens))
        expected_y = byte_tokens + [EOS_ID] + [PAD_ID] * (context - len(byte_tokens) - 1)
        actual_x = binary[example_index, 0].tolist()
        actual_y = binary[example_index, 1].tolist()
        if actual_x != expected_x or actual_y != expected_y:
            raise ValueError(f"source/binary token mismatch at example {example_index}")
    return {
        "verified_examples": len(records),
        "mismatches": 0,
        "duration_ns": now_ns() - started,
        "exact_x_y_masks_verified": True,
    }


def load_schedule(root: Path, schedule_info: Mapping[str, Any], seed: int, lengths: Sequence[int]) -> Tuple[List[Batch], Dict[str, Any]]:
    path = root / schedule_info["relative_path"]
    expected_sha = str(schedule_info["binary_payload_sha256"])
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"schedule SHA mismatch: {path}")
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if int(schedule["seed"]) != seed:
        raise ValueError("schedule seed mismatch")
    if int(schedule["total_examples"]) != len(lengths):
        raise ValueError("schedule example count mismatch")
    flat: List[int] = []
    batches: List[Batch] = []
    for group in schedule["groups"]:
        indices = tuple(int(index) for index in group["indices"])
        if not indices:
            raise ValueError("empty schedule group")
        source_bytes = sum(int(lengths[index]) for index in indices)
        if source_bytes != int(group["source_bytes"]):
            raise ValueError(f"schedule source byte mismatch at step {group['step']}")
        if int(group["raw_byte_target_tokens"]) != source_bytes:
            raise ValueError("raw target mismatch")
        if int(group["nonpad_target_tokens"]) != source_bytes + len(indices):
            raise ValueError("non-pad target mismatch")
        microbatch_size = int(group["microbatch_size"])
        for microbatch_index, start in enumerate(range(0, len(indices), microbatch_size)):
            batch_indices = indices[start : start + microbatch_size]
            batch_source_bytes = sum(int(lengths[index]) for index in batch_indices)
            batches.append(
                Batch(
                    seed=seed,
                    step=int(group["step"]),
                    microbatch_index=microbatch_index,
                    indices=batch_indices,
                    source_bytes=batch_source_bytes,
                    nonpad_target_tokens=batch_source_bytes + len(batch_indices),
                    raw_byte_target_tokens=batch_source_bytes,
                )
            )
        flat.extend(indices)
    if sorted(flat) != list(range(len(lengths))):
        raise ValueError("schedule is not a single exact permutation")
    if int(schedule["num_steps"]) != len(schedule["groups"]):
        raise ValueError("schedule step count mismatch")
    return batches, {
        "path": str(path),
        "sha256": actual_sha,
        "seed": seed,
        "steps": len(schedule["groups"]),
        "batches": len(batches),
        "examples": len(flat),
        "source_bytes": sum(batch.source_bytes for batch in batches),
        "permutation_sha256": sha256_bytes(",".join(str(index) for index in flat).encode("ascii")),
    }


def make_event(
    *,
    run_id: str,
    seed: int,
    arm: str,
    repetition: int,
    batch_number: int,
    batch: Batch,
    phase: str,
    duration_ns: int,
    queue_wait_ns: int,
    inflight: int,
    max_inflight: int,
    worker_count: int,
    worker_name: str,
    fingerprints: Mapping[str, str],
    index_hash: str,
    payload_bytes: int,
    cpu_only: bool = True,
    **extra: Any,
) -> Dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "cpu_only": cpu_only,
        "run_id": run_id,
        "seed": seed,
        "arm": arm,
        "repetition": repetition,
        "batch_number": batch_number,
        "step": batch.step,
        "microbatch_index": batch.microbatch_index,
        "indices_count": len(batch.indices),
        "index_sha256": index_hash,
        "source_bytes": batch.source_bytes,
        "nonpad_target_tokens": batch.nonpad_target_tokens,
        "raw_byte_target_tokens": batch.raw_byte_target_tokens,
        "payload_bytes": payload_bytes,
        "phase": phase,
        "duration_ns": int(duration_ns),
        "queue_wait_ns": int(queue_wait_ns),
        "inflight": int(inflight),
        "max_inflight": int(max_inflight),
        "worker_count": int(worker_count),
        "worker_name": worker_name,
        "x_sha256": fingerprints["x_sha256"],
        "y_sha256": fingerprints["y_sha256"],
        "valid_target_mask_sha256": fingerprints["valid_target_mask_sha256"],
        "raw_target_mask_sha256": fingerprints["raw_target_mask_sha256"],
        "fingerprint_sha256": fingerprints["fingerprint_sha256"],
        "valid_target_count": int(fingerprints["valid_target_count"]),
        "raw_target_count": int(fingerprints["raw_target_count"]),
    }
    result.update(extra)
    return result


def assemble_current(raw: torch.Tensor, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    x_items = [raw[index, 0].clone() for index in batch.indices]
    y_items = [raw[index, 1].clone() for index in batch.indices]
    return torch.stack(x_items), torch.stack(y_items), 0, 0


def assemble_indexed(precomputed: torch.Tensor, batch: Batch, index_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    selected = precomputed.index_select(0, index_tensor)
    return selected[:, 0].contiguous(), selected[:, 1].contiguous(), 0, 0


def run_direct_arm(
    *,
    arm: str,
    seed: int,
    repetition: int,
    batches: Sequence[Batch],
    raw: torch.Tensor,
    precomputed: torch.Tensor,
    index_tensors: Sequence[torch.Tensor],
    trace_handle: Any,
    assembly_fn: Any,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    run_id = f"ssre_npa_v3_seed{seed}_{arm}_rep{repetition}"
    digests: Dict[int, str] = {}
    total_start = now_ns()
    total_assembly = 0
    total_hash = 0
    total_source = 0
    max_batch_payload = 0
    for batch_number, batch in enumerate(batches):
        started = now_ns()
        if arm == "current_python_stack":
            x, y, _, _ = assembly_fn(raw, batch)
        else:
            x, y, _, _ = assembly_fn(precomputed, batch, index_tensors[batch_number])
        assembled = now_ns()
        fingerprints = batch_fingerprint(x, y)
        finished = now_ns()
        if int(fingerprints["valid_target_count"]) != batch.nonpad_target_tokens:
            raise ValueError(f"valid target count mismatch in {run_id}:{batch_number}")
        if int(fingerprints["raw_target_count"]) != batch.raw_byte_target_tokens:
            raise ValueError(f"raw target count mismatch in {run_id}:{batch_number}")
        index_hash = index_digest(batch.indices)
        digests[batch_number] = fingerprints["fingerprint_sha256"]
        assembly_ns = assembled - started
        hash_ns = finished - assembled
        payload_bytes = (x.numel() + y.numel()) * x.element_size()
        total_assembly += assembly_ns
        total_hash += hash_ns
        total_source += batch.source_bytes
        max_batch_payload = max(max_batch_payload, payload_bytes)
        trace_handle.write(
            json.dumps(
                make_event(
                    run_id=run_id,
                    seed=seed,
                    arm=arm,
                    repetition=repetition,
                    batch_number=batch_number,
                    batch=batch,
                    phase="commit",
                    duration_ns=finished - started,
                    queue_wait_ns=0,
                    inflight=0,
                    max_inflight=0,
                    worker_count=0,
                    worker_name="main",
                    fingerprints=fingerprints,
                    index_hash=index_hash,
                    payload_bytes=payload_bytes,
                    assembly_ns=assembly_ns,
                    hash_ns=hash_ns,
                ),
                sort_keys=True,
            )
            + "\n"
        )
    total_ns = now_ns() - total_start
    return {
        "run_id": run_id,
        "seed": seed,
        "arm": arm,
        "repetition": repetition,
        "batches": len(batches),
        "examples": sum(len(batch.indices) for batch in batches),
        "source_bytes": total_source,
        "wall_ns": total_ns,
        "assembly_ns": total_assembly,
        "hash_ns": total_hash,
        "batches_per_second": len(batches) / (total_ns / 1e9),
        "examples_per_second": sum(len(batch.indices) for batch in batches) / (total_ns / 1e9),
        "source_bytes_per_second": total_source / (total_ns / 1e9),
        "max_batch_payload_bytes": max_batch_payload,
        "max_inflight": 0,
        "worker_prep_ns": total_assembly,
        "queue_wait_ns": 0,
        "integrity": {"fingerprint_count": len(digests), "mismatches": 0},
    }, digests


def reservoir_job(precomputed: torch.Tensor, batch: Batch, index_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
    started = now_ns()
    selected = precomputed.index_select(0, index_tensor)
    x = selected[:, 0].contiguous()
    y = selected[:, 1].contiguous()
    finished = now_ns()
    return x, y, started, finished, threading.current_thread().name


def run_reservoir(
    *,
    seed: int,
    repetition: int,
    batches: Sequence[Batch],
    precomputed: torch.Tensor,
    index_tensors: Sequence[torch.Tensor],
    workers: int,
    window: int,
    trace_handle: Any,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    arm = "cpu_reservoir"
    run_id = f"ssre_npa_v3_seed{seed}_{arm}_rep{repetition}"
    digests: Dict[int, str] = {}
    pending: Dict[int, Tuple[Future[Any], int]] = {}
    next_submit = 0
    next_commit = 0
    total_source = 0
    total_assembly = 0
    total_hash = 0
    total_queue_wait = 0
    max_inflight = 0
    max_batch_payload = 0
    total_start = now_ns()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssre-npa-v3") as pool:
        while next_commit < len(batches):
            while next_submit < len(batches) and len(pending) < window:
                submitted = now_ns()
                pending[next_submit] = (
                    pool.submit(reservoir_job, precomputed, batches[next_submit], index_tensors[next_submit]),
                    submitted,
                )
                next_submit += 1
                max_inflight = max(max_inflight, len(pending))
            future, submitted = pending.pop(next_commit)
            wait_started = now_ns()
            x, y, worker_started, worker_finished, worker_name = future.result()
            committed = now_ns()
            queue_wait = max(0, committed - wait_started)
            fingerprints = batch_fingerprint(x, y)
            hashed = now_ns()
            batch = batches[next_commit]
            if int(fingerprints["valid_target_count"]) != batch.nonpad_target_tokens:
                raise ValueError(f"valid target count mismatch in {run_id}:{next_commit}")
            if int(fingerprints["raw_target_count"]) != batch.raw_byte_target_tokens:
                raise ValueError(f"raw target count mismatch in {run_id}:{next_commit}")
            index_hash = index_digest(batch.indices)
            digests[next_commit] = fingerprints["fingerprint_sha256"]
            worker_prep = worker_finished - worker_started
            hash_ns = hashed - committed
            payload_bytes = (x.numel() + y.numel()) * x.element_size()
            total_source += batch.source_bytes
            total_assembly += worker_prep
            total_hash += hash_ns
            total_queue_wait += queue_wait
            max_batch_payload = max(max_batch_payload, payload_bytes)
            trace_handle.write(
                json.dumps(
                    make_event(
                        run_id=run_id,
                        seed=seed,
                        arm=arm,
                        repetition=repetition,
                        batch_number=next_commit,
                        batch=batch,
                        phase="commit",
                        duration_ns=committed - submitted,
                        queue_wait_ns=queue_wait,
                        inflight=len(pending),
                        max_inflight=max_inflight,
                        worker_count=workers,
                        worker_name=worker_name,
                        fingerprints=fingerprints,
                        index_hash=index_hash,
                        payload_bytes=payload_bytes,
                        assembly_ns=worker_prep,
                        hash_ns=hash_ns,
                        dispatch_to_commit_ns=committed - submitted,
                        worker_started_ns=worker_started,
                        worker_finished_ns=worker_finished,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            next_commit += 1
    total_ns = now_ns() - total_start
    examples = sum(len(batch.indices) for batch in batches)
    return {
        "run_id": run_id,
        "seed": seed,
        "arm": arm,
        "repetition": repetition,
        "batches": len(batches),
        "examples": examples,
        "source_bytes": total_source,
        "wall_ns": total_ns,
        "assembly_ns": total_assembly,
        "hash_ns": total_hash,
        "batches_per_second": len(batches) / (total_ns / 1e9),
        "examples_per_second": examples / (total_ns / 1e9),
        "source_bytes_per_second": total_source / (total_ns / 1e9),
        "max_batch_payload_bytes": max_batch_payload,
        "max_inflight": max_inflight,
        "worker_prep_ns": total_assembly,
        "queue_wait_ns": total_queue_wait,
        "workers": workers,
        "window": window,
        "integrity": {"fingerprint_count": len(digests), "mismatches": 0},
    }, digests


def environment_report() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_version": torch.__version__,
        "torch_num_threads_before": torch.get_num_threads(),
        "cuda_profiled": False,
        "gpu_measurement": "unavailable_or_not_attempted",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only measured Crucible batch replay")
    parser.add_argument("--context-root", type=Path, default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,1337")
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--timed-repetitions", type=int, default=2)
    parser.add_argument("--reservoir-workers", type=int, default=2)
    parser.add_argument("--reservoir-window", type=int, default=8)
    args = parser.parse_args()
    if args.warmup_repetitions < 1 or args.timed_repetitions < 2:
        raise ValueError("use at least one warmup and two timed repetitions")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    root = args.context_root.resolve()
    manifest_base = root / "crucible_manifest"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, config_sha = load_manifest(root)
    hp = config["hyperparameters"]
    context = int(hp["context_window_length"])
    if context != 128:
        raise ValueError("v3 default replay is intentionally context-128 only")
    if int(hp["max_source_utf8_bytes"]) != context - 1:
        raise ValueError("context/source geometry mismatch")
    torch.set_num_threads(1)
    environment = environment_report()
    environment["torch_num_threads_after"] = torch.get_num_threads()
    train_info = config["splits"]["train"]
    # Frozen manifest paths are relative to its own directory, matching the
    # actual Crucible runner's base_dir = dirname(config_path) contract.
    records, source_report = load_source_records(manifest_base, train_info)
    raw, binary_report = load_binary(manifest_base, train_info, len(records), context)
    source_binary_report = verify_source_binary(records, raw, context)
    precompute_started = now_ns()
    precomputed = raw.clone().contiguous()
    precompute_report = {
        "duration_ns": now_ns() - precompute_started,
        "bytes": precomputed.numel() * precomputed.element_size(),
        "peak_rss_bytes_after": current_rss_bytes(),
    }

    trace_path = output / "measured_batch_trace.jsonl"
    metrics: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    null_trace = NullTrace()
    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for seed in seeds:
            schedule_key = str(seed)
            schedule_info = config["schedules"][schedule_key]
            batches, schedule_report = load_schedule(manifest_base, schedule_info, seed, [len(item) for item in records])
            index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
            for repetition in range(1, args.warmup_repetitions + 1):
                run_direct_arm(
                    arm="current_python_stack",
                    seed=seed,
                    repetition=-repetition,
                    batches=batches,
                    raw=raw,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    trace_handle=null_trace,
                    assembly_fn=assemble_current,
                )
                run_direct_arm(
                    arm="indexed_precomputed",
                    seed=seed,
                    repetition=-repetition,
                    batches=batches,
                    raw=raw,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    trace_handle=null_trace,
                    assembly_fn=assemble_indexed,
                )
                run_reservoir(
                    seed=seed,
                    repetition=-repetition,
                    batches=batches,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    workers=args.reservoir_workers,
                    window=args.reservoir_window,
                    trace_handle=null_trace,
                )
            baseline_digests_by_rep: Dict[int, Dict[int, str]] = {}
            for repetition in range(1, args.timed_repetitions + 1):
                baseline_metrics, baseline_digests = run_direct_arm(
                    arm="current_python_stack",
                    seed=seed,
                    repetition=repetition,
                    batches=batches,
                    raw=raw,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    trace_handle=trace_handle,
                    assembly_fn=assemble_current,
                )
                indexed_metrics, indexed_digests = run_direct_arm(
                    arm="indexed_precomputed",
                    seed=seed,
                    repetition=repetition,
                    batches=batches,
                    raw=raw,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    trace_handle=trace_handle,
                    assembly_fn=assemble_indexed,
                )
                reservoir_metrics, reservoir_digests = run_reservoir(
                    seed=seed,
                    repetition=repetition,
                    batches=batches,
                    precomputed=precomputed,
                    index_tensors=index_tensors,
                    workers=args.reservoir_workers,
                    window=args.reservoir_window,
                    trace_handle=trace_handle,
                )
                for candidate_name, candidate_digests in (
                    ("indexed_precomputed", indexed_digests),
                    ("cpu_reservoir", reservoir_digests),
                ):
                    mismatches = [
                        batch_number
                        for batch_number in baseline_digests
                        if baseline_digests[batch_number] != candidate_digests.get(batch_number)
                    ]
                    comparisons.append(
                        {
                            "seed": seed,
                            "repetition": repetition,
                            "candidate": candidate_name,
                            "reference": "current_python_stack",
                            "batch_count": len(batches),
                            "digest_mismatches": mismatches,
                            "exact_equivalence": not mismatches,
                        }
                    )
                for item in (baseline_metrics, indexed_metrics, reservoir_metrics):
                    item["schedule"] = schedule_report
                    item["config_sha256"] = config_sha
                    item["peak_rss_bytes"] = current_rss_bytes()
                    metrics.append(item)
    summary = {
        "schema_version": "ssre-npa-v3-summary",
        "source_kind": SOURCE_KIND,
        "cpu_only": True,
        "cuda_profiled": False,
        "context_root": str(root),
        "manifest_config_sha256": config_sha,
        "environment": environment,
        "source_report": source_report,
        "binary_report": binary_report,
        "source_binary_report": source_binary_report,
        "precompute_report": precompute_report,
        "parameters": {
            "seeds": seeds,
            "warmup_repetitions": args.warmup_repetitions,
            "timed_repetitions": args.timed_repetitions,
            "reservoir_workers": args.reservoir_workers,
            "reservoir_window": args.reservoir_window,
            "context": context,
            "microbatch_size": int(hp["microbatch_size"]),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "integrity": {
            "all_candidate_digests_match_reference": all(item["exact_equivalence"] for item in comparisons),
            "all_source_binary_tokens_and_masks_verified": bool(source_binary_report["exact_x_y_masks_verified"]),
            "schedule_count": len(seeds),
        },
        "measurement_limitations": [
            "This gate measures CPU source/batch delivery only; no GPU forward/backward or H2D time was measured.",
            "The frozen binary is already tokenized; source JSONL validation is measured separately from per-batch delivery.",
            "Thread-pool reservoir timing is a CPU delivery measurement, not proof that a CUDA consumer will overlap successfully.",
        ],
    }
    (output / "measured_batch_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "trace": str(trace_path),
        "metrics": len(metrics),
        "comparisons": len(comparisons),
        "equivalence": summary["integrity"]["all_candidate_digests_match_reference"],
        "cuda_profiled": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

