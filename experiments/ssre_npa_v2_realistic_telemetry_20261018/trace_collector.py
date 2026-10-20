#!/usr/bin/env python3
"""Deterministic virtual workload telemetry for SSRE-NPA v2.

This module intentionally has no Torch/CUDA dependency and performs no live
training. It provides the event contract and a deterministic workload proxy so
schema coverage, replay and bounded-resource accounting can be tested before a
real scheduler adapter is attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "ssre-npa-telemetry-v2"
SOURCE_KIND = "deterministic_virtual_workload_proxy"
PHASES = {
    "dispatch",
    "prepare",
    "queue_wait",
    "stage",
    "h2d",
    "gpu_compute",
    "commit",
    "backpressure",
    "scheduler_decision",
}


@dataclass(frozen=True)
class Packet:
    seq: int
    payload_bytes: int
    token_count: int
    prep_ms: float
    gpu_ms: float
    cache_state: str
    telemetry_ms: float


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def choose_weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    mark = rng.random() * sum(float(value) for value in weights.values())
    running = 0.0
    for key, value in weights.items():
        running += float(value)
        if mark <= running:
            return key
    return next(reversed(weights))


def make_packets(scenario: Mapping[str, Any], count: int) -> List[Packet]:
    rng = random.Random(int(scenario["seed"]))
    payload_low, payload_high = (int(value) for value in scenario["payload_bytes"])
    prep_low, prep_high = (float(value) for value in scenario["prep_ms"])
    gpu_low, gpu_high = (float(value) for value in scenario["gpu_ms"])
    cache_weights = {str(key): float(value) for key, value in scenario["cache_states"].items()}
    packets: List[Packet] = []
    cache_multiplier = {"hot": 0.7, "warm": 1.25, "cold": 3.5}
    for seq in range(count):
        payload = rng.randint(payload_low, payload_high)
        cache_state = choose_weighted(rng, cache_weights)
        prep = rng.uniform(prep_low, prep_high) * cache_multiplier.get(cache_state, 1.0)
        if rng.random() < float(scenario["straggler_fraction"]):
            prep *= 6.0
        telemetry = float(scenario["telemetry_factor"]) * (0.35 + payload / 262144.0)
        gpu = rng.uniform(gpu_low, gpu_high)
        packets.append(
            Packet(
                seq=seq,
                payload_bytes=payload,
                token_count=max(1, payload),
                prep_ms=prep,
                gpu_ms=gpu,
                cache_state=cache_state,
                telemetry_ms=telemetry,
            )
        )
    return packets


class TraceCollector:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._event_id = 0

    def emit(
        self,
        *,
        run_id: str,
        scenario_id: str,
        scheduler_arm: str,
        packet_seq: int | None,
        worker_id: int | None,
        phase: str,
        start_ms: float,
        duration_ms: float,
        payload_bytes: int,
        token_count: int,
        cache_state: str,
        queue_depth: int,
        inflight_packets: int,
        inflight_bytes: int,
        worker_count: int,
        window_packets: int,
        window_bytes: int,
        outcome: str = "ok",
        **extra: Any,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        event = {
            "schema_version": SCHEMA_VERSION,
            "source_kind": SOURCE_KIND,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scheduler_arm": scheduler_arm,
            "event_id": self._event_id,
            "packet_seq": packet_seq,
            "worker_id": worker_id,
            "phase": phase,
            "start_ms": float(start_ms),
            "duration_ms": float(duration_ms),
            "payload_bytes": int(payload_bytes),
            "token_count": int(token_count),
            "cache_state": cache_state,
            "queue_depth": int(queue_depth),
            "inflight_packets": int(inflight_packets),
            "inflight_bytes": int(inflight_bytes),
            "worker_count": int(worker_count),
            "window_packets": int(window_packets),
            "window_bytes": int(window_bytes),
            "outcome": outcome,
        }
        event.update(extra)
        self.events.append(event)
        self._event_id += 1


def simulate_arm(
    collector: TraceCollector,
    scenario: Mapping[str, Any],
    arm_name: str,
    arm: Mapping[str, Any],
    packets: Sequence[Packet],
    window_packets: int,
    window_bytes: int,
) -> Dict[str, Any]:
    scenario_id = str(scenario["id"])
    run_id = f"ssre_npa_v2_{scenario_id}_{arm_name}"
    max_workers = int(arm["workers"])
    queue_capacity = int(arm["queue_capacity"])
    preparation_factor = float(arm["preparation_factor"])
    worker_free = [0.0 for _ in range(max_workers)]
    inflight: Dict[int, Tuple[Packet, int, float, float]] = {}
    worker_clock = 0.0
    gpu_free = 0.0
    committed = 0
    next_seq = 0
    active_workers = max_workers
    recent_queue_wait: List[float] = []
    local_start_event = len(collector.events)

    while committed < len(packets):
        if bool(arm.get("adaptive")) and committed % 16 == 0:
            mean_wait = sum(recent_queue_wait[-8:]) / max(1, len(recent_queue_wait[-8:]))
            if mean_wait > 2.0:
                active_workers = min(max_workers, active_workers + 1)
            elif mean_wait < 0.25 and active_workers > 2:
                active_workers -= 1
            collector.emit(
                run_id=run_id,
                scenario_id=scenario_id,
                scheduler_arm=arm_name,
                packet_seq=None,
                worker_id=None,
                phase="scheduler_decision",
                start_ms=worker_clock,
                duration_ms=0.01,
                payload_bytes=0,
                token_count=0,
                cache_state="not_applicable",
                queue_depth=len(inflight),
                inflight_packets=len(inflight),
                inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
                worker_count=active_workers,
                window_packets=window_packets,
                window_bytes=window_bytes,
                predicted_queue_empty_ms=max(0.0, mean_wait),
                observed_queue_empty_ms=max(0.0, mean_wait),
                reason="virtual_heuristic_decision",
            )

        while next_seq < len(packets) and len(inflight) < window_packets:
            packet = packets[next_seq]
            current_bytes = sum(item[0].payload_bytes for item in inflight.values())
            if current_bytes + packet.payload_bytes > window_bytes:
                break
            eligible = worker_free[:active_workers]
            worker_id = min(range(active_workers), key=lambda index: eligible[index])
            dispatch_at = max(worker_clock, worker_free[worker_id])
            worker_clock = dispatch_at
            scaled_prep = packet.prep_ms * preparation_factor + packet.telemetry_ms
            prep_end = dispatch_at + scaled_prep
            worker_free[worker_id] = prep_end
            inflight[next_seq] = (packet, worker_id, dispatch_at, prep_end)
            inflight_count = len(inflight)
            inflight_payload = current_bytes + packet.payload_bytes
            collector.emit(
                run_id=run_id,
                scenario_id=scenario_id,
                scheduler_arm=arm_name,
                packet_seq=packet.seq,
                worker_id=worker_id,
                phase="dispatch",
                start_ms=dispatch_at,
                duration_ms=0.01,
                payload_bytes=packet.payload_bytes,
                token_count=packet.token_count,
                cache_state=packet.cache_state,
                queue_depth=inflight_count,
                inflight_packets=inflight_count,
                inflight_bytes=inflight_payload,
                worker_count=active_workers,
                window_packets=window_packets,
                window_bytes=window_bytes,
                packet_digest=hashlib.sha256(f"{run_id}:{packet.seq}".encode()).hexdigest(),
            )
            collector.emit(
                run_id=run_id,
                scenario_id=scenario_id,
                scheduler_arm=arm_name,
                packet_seq=packet.seq,
                worker_id=worker_id,
                phase="prepare",
                start_ms=dispatch_at,
                duration_ms=scaled_prep,
                payload_bytes=packet.payload_bytes,
                token_count=packet.token_count,
                cache_state=packet.cache_state,
                queue_depth=inflight_count,
                inflight_packets=inflight_count,
                inflight_bytes=inflight_payload,
                worker_count=active_workers,
                window_packets=window_packets,
                window_bytes=window_bytes,
            )
            next_seq += 1

        if next_seq < len(packets) and len(inflight) >= window_packets:
            next_packet = packets[next_seq]
            current_packet, _, _, current_prep_end = inflight[committed]
            release_start = max(gpu_free, current_prep_end)
            release_duration = (
                0.04
                + current_packet.payload_bytes / 2_000_000.0
                + 0.03
                + current_packet.payload_bytes / 12_000_000.0
                + current_packet.gpu_ms
                + 0.01
            )
            collector.emit(
                run_id=run_id,
                scenario_id=scenario_id,
                scheduler_arm=arm_name,
                packet_seq=next_packet.seq,
                worker_id=None,
                phase="backpressure",
                start_ms=worker_clock,
                duration_ms=max(0.0, release_start + release_duration - worker_clock),
                payload_bytes=next_packet.payload_bytes,
                token_count=next_packet.token_count,
                cache_state=next_packet.cache_state,
                queue_depth=len(inflight),
                inflight_packets=len(inflight),
                inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
                worker_count=active_workers,
                window_packets=window_packets,
                window_bytes=window_bytes,
                outcome="blocked",
                reason="global_inflight_packet_window",
            )

        packet, worker_id, dispatch_at, prep_end = inflight[committed]
        queue_wait = max(0.0, gpu_free - prep_end)
        stage_start = max(gpu_free, prep_end)
        stage_ms = 0.04 + packet.payload_bytes / 2_000_000.0
        h2d_ms = 0.03 + packet.payload_bytes / 12_000_000.0
        gpu_start = stage_start + stage_ms + h2d_ms
        commit_at = gpu_start + packet.gpu_ms
        recent_queue_wait.append(queue_wait)
        remaining_bytes = sum(item[0].payload_bytes for seq, item in inflight.items() if seq != committed)
        remaining_packets = len(inflight) - 1
        collector.emit(
            run_id=run_id,
            scenario_id=scenario_id,
            scheduler_arm=arm_name,
            packet_seq=packet.seq,
            worker_id=worker_id,
            phase="queue_wait",
            start_ms=prep_end,
            duration_ms=queue_wait,
            payload_bytes=packet.payload_bytes,
            token_count=packet.token_count,
            cache_state=packet.cache_state,
            queue_depth=len(inflight),
            inflight_packets=len(inflight),
            inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
            worker_count=active_workers,
            window_packets=window_packets,
            window_bytes=window_bytes,
            gpu_wait_ms=queue_wait,
        )
        collector.emit(
            run_id=run_id,
            scenario_id=scenario_id,
            scheduler_arm=arm_name,
            packet_seq=packet.seq,
            worker_id=worker_id,
            phase="stage",
            start_ms=stage_start,
            duration_ms=stage_ms,
            payload_bytes=packet.payload_bytes,
            token_count=packet.token_count,
            cache_state=packet.cache_state,
            queue_depth=len(inflight),
            inflight_packets=len(inflight),
            inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
            worker_count=active_workers,
            window_packets=window_packets,
            window_bytes=window_bytes,
        )
        collector.emit(
            run_id=run_id,
            scenario_id=scenario_id,
            scheduler_arm=arm_name,
            packet_seq=packet.seq,
            worker_id=worker_id,
            phase="h2d",
            start_ms=stage_start + stage_ms,
            duration_ms=h2d_ms,
            payload_bytes=packet.payload_bytes,
            token_count=packet.token_count,
            cache_state=packet.cache_state,
            queue_depth=len(inflight),
            inflight_packets=len(inflight),
            inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
            worker_count=active_workers,
            window_packets=window_packets,
            window_bytes=window_bytes,
        )
        collector.emit(
            run_id=run_id,
            scenario_id=scenario_id,
            scheduler_arm=arm_name,
            packet_seq=packet.seq,
            worker_id=worker_id,
            phase="gpu_compute",
            start_ms=gpu_start,
            duration_ms=packet.gpu_ms,
            payload_bytes=packet.payload_bytes,
            token_count=packet.token_count,
            cache_state=packet.cache_state,
            queue_depth=len(inflight),
            inflight_packets=len(inflight),
            inflight_bytes=sum(item[0].payload_bytes for item in inflight.values()),
            worker_count=active_workers,
            window_packets=window_packets,
            window_bytes=window_bytes,
        )
        collector.emit(
            run_id=run_id,
            scenario_id=scenario_id,
            scheduler_arm=arm_name,
            packet_seq=packet.seq,
            worker_id=worker_id,
            phase="commit",
            start_ms=commit_at,
            duration_ms=0.01,
            payload_bytes=packet.payload_bytes,
            token_count=packet.token_count,
            cache_state=packet.cache_state,
            queue_depth=remaining_packets,
            inflight_packets=remaining_packets,
            inflight_bytes=remaining_bytes,
            worker_count=active_workers,
            window_packets=window_packets,
            window_bytes=window_bytes,
            packet_digest=hashlib.sha256(f"{run_id}:{packet.seq}".encode()).hexdigest(),
        )
        del inflight[committed]
        committed += 1
        worker_clock = max(worker_clock, commit_at)
        gpu_free = commit_at

    run_events = collector.events[local_start_event:]
    commit_events = [event for event in run_events if event["phase"] == "commit"]
    queue_waits = [event["duration_ms"] for event in run_events if event["phase"] == "queue_wait"]
    gpu_events = [event for event in run_events if event["phase"] == "gpu_compute"]
    prep_events = [event for event in run_events if event["phase"] == "prepare"]
    backpressure = [event for event in run_events if event["phase"] == "backpressure"]
    wall_ms = max((float(event["start_ms"]) + float(event["duration_ms"]) for event in run_events), default=0.0)
    payload = sum(int(event["payload_bytes"]) for event in commit_events)
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "scheduler_arm": arm_name,
        "source_kind": SOURCE_KIND,
        "packet_count": len(commit_events),
        "payload_bytes": payload,
        "virtual_wall_ms": wall_ms,
        "packets_per_second_virtual": len(commit_events) / (wall_ms / 1000.0) if wall_ms else 0.0,
        "payload_mb_per_second_virtual": payload / (wall_ms / 1000.0) / 1_000_000.0 if wall_ms else 0.0,
        "prepare_cpu_ms": sum(float(event["duration_ms"]) for event in prep_events),
        "gpu_compute_ms": sum(float(event["duration_ms"]) for event in gpu_events),
        "queue_wait_ms_total": sum(queue_waits),
        "queue_wait_ms_p50": percentile(queue_waits, 0.50),
        "queue_wait_ms_p95": percentile(queue_waits, 0.95),
        "backpressure_events": len(backpressure),
        "backpressure_blocked_ms": sum(float(event["duration_ms"]) for event in backpressure),
        "max_inflight_packets": max((int(event["inflight_packets"]) for event in run_events), default=0),
        "max_inflight_bytes": max((int(event["inflight_bytes"]) for event in run_events), default=0),
        "max_queue_depth": max((int(event["queue_depth"]) for event in run_events), default=0),
        "scheduler_decisions": sum(event["phase"] == "scheduler_decision" for event in run_events),
        "worker_count_max": max((int(event["worker_count"]) for event in run_events), default=0),
        "integrity": {
            "commits_in_order": [event["packet_seq"] for event in commit_events] == list(range(len(packets))),
            "commit_count": len(commit_events),
            "expected_commit_count": len(packets),
        },
    }


def validate_events(events: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any]) -> Dict[str, Any]:
    if not events:
        raise ValueError("trace is empty")
    expected_ids = list(range(len(events)))
    actual_ids = [int(event["event_id"]) for event in events]
    if actual_ids != expected_ids:
        raise ValueError("event_id is not strictly contiguous")
    window_packets = int(matrix["global_window_packets"])
    window_bytes = int(matrix["global_window_bytes"])
    by_run: Dict[str, List[Mapping[str, Any]]] = {}
    for event in events:
        if event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema version mismatch")
        if event.get("source_kind") != SOURCE_KIND:
            raise ValueError("unexpected source kind")
        for field in ("start_ms", "duration_ms"):
            if not math.isfinite(float(event[field])) or float(event[field]) < 0:
                raise ValueError(f"invalid {field}")
        for field in ("payload_bytes", "token_count", "queue_depth", "inflight_packets", "inflight_bytes"):
            if int(event[field]) < 0:
                raise ValueError(f"negative {field}")
        if int(event["inflight_packets"]) > window_packets:
            raise ValueError("packet window exceeded")
        if int(event["inflight_bytes"]) > window_bytes:
            raise ValueError("byte window exceeded")
        packet_seq = event.get("packet_seq")
        if packet_seq is not None and int(packet_seq) < 0:
            raise ValueError("negative packet sequence")
        by_run.setdefault(str(event["run_id"]), []).append(event)
    run_checks: Dict[str, Any] = {}
    for run_id, run_events in by_run.items():
        commits = [int(event["packet_seq"]) for event in run_events if event["phase"] == "commit"]
        dispatches = {int(event["packet_seq"]) for event in run_events if event["phase"] == "dispatch"}
        if commits != list(range(len(commits))):
            raise ValueError(f"commit order failure in {run_id}")
        if set(commits) != dispatches:
            raise ValueError(f"dispatch/commit mismatch in {run_id}")
        run_checks[run_id] = {
            "commit_count": len(commits),
            "commits_in_order": True,
            "dispatch_commit_sets_match": True,
        }
    return {
        "event_count": len(events),
        "run_count": len(by_run),
        "run_checks": run_checks,
        "global_window_packets": window_packets,
        "global_window_bytes": window_bytes,
        "all_invariants_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/validate SSRE-NPA v2 virtual telemetry")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    collector = TraceCollector()
    summaries: List[Dict[str, Any]] = []
    packet_count = int(matrix["packet_count"])
    for scenario in matrix["scenarios"]:
        packets = make_packets(scenario, packet_count)
        for arm_name, arm in matrix["scheduler_arms"].items():
            summaries.append(
                simulate_arm(
                    collector,
                    scenario,
                    str(arm_name),
                    arm,
                    packets,
                    int(matrix["global_window_packets"]),
                    int(matrix["global_window_bytes"]),
                )
            )
    validation = validate_events(collector.events, matrix)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in collector.events),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "ssre-npa-summary-v2",
        "source_kind": SOURCE_KIND,
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "neural_controller_training": False,
        "measured_hardware": False,
        "validation": validation,
        "runs": summaries,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(collector.events), "runs": len(summaries), "invariants": validation["all_invariants_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

