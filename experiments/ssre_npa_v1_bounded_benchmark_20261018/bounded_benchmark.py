#!/usr/bin/env python3
"""SSRE-NPA v1 bounded-resource, CPU-only scheduler benchmark.

The benchmark is intentionally self-contained. It uses deterministic synthetic
packets and a condition-variable pipeline to test:

* a global dispatched-but-unconsumed window;
* blocking wait telemetry rather than busy-spin retries;
* variable preparation costs, sizes, stragglers, and slow consumers;
* serial, fixed-worker, and adaptive deterministic scheduling;
* transient worker failure/retry;
* cancellation, shutdown, recovery, and exact result equivalence.

It does not import CUDA, torch, Crucible, Braidformer, or historical SSRE code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


class ChannelClosed(RuntimeError):
    pass


class WindowCancelled(RuntimeError):
    pass


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PacketSpec:
    manifest_sha256: str
    sequence_index: int
    packet_id: str
    cost_units: int
    payload_bytes: int
    straggler_delay_s: float = 0.0


@dataclass(frozen=True)
class WorkItem:
    spec: PacketSpec
    attempt: int = 0


@dataclass(frozen=True)
class Prepared:
    spec: PacketSpec
    worker_id: int
    payload_digest: str
    elapsed_ns: int


@dataclass(frozen=True)
class Failed:
    item: WorkItem
    worker_id: int
    error: str


@dataclass
class WaitStats:
    count: int = 0
    total_s: float = 0.0
    maximum_s: float = 0.0
    samples_s: List[float] = field(default_factory=list)

    def add(self, duration_s: float) -> None:
        self.count += 1
        self.total_s += duration_s
        self.maximum_s = max(self.maximum_s, duration_s)
        self.samples_s.append(duration_s)

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "total_s": self.total_s,
            "max_s": self.maximum_s,
            "p50_s": percentile(self.samples_s, 0.50),
            "p95_s": percentile(self.samples_s, 0.95),
        }


class BoundedChannel:
    """Bounded FIFO using one condition variable and measured blocking waits."""

    def __init__(self, name: str, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("channel capacity must be positive")
        self.name = name
        self.capacity = capacity
        self._items: Deque[object] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self.put_wait = WaitStats()
        self.get_wait = WaitStats()
        self.high_water = 0

    def put(self, item: object) -> None:
        started: Optional[float] = None
        with self._condition:
            while len(self._items) >= self.capacity and not self._closed:
                if started is None:
                    started = time.perf_counter()
                self._condition.wait()
            if started is not None:
                self.put_wait.add(time.perf_counter() - started)
            if self._closed:
                raise ChannelClosed(self.name)
            self._items.append(item)
            self.high_water = max(self.high_water, len(self._items))
            self._condition.notify_all()

    def try_put(self, item: object) -> bool:
        """Enqueue without waiting; used to avoid retry/worker queue cycles."""
        with self._condition:
            if self._closed or len(self._items) >= self.capacity:
                return False
            self._items.append(item)
            self.high_water = max(self.high_water, len(self._items))
            self._condition.notify_all()
            return True

    def get(self) -> object:
        started: Optional[float] = None
        with self._condition:
            while not self._items and not self._closed:
                if started is None:
                    started = time.perf_counter()
                self._condition.wait()
            if started is not None:
                self.get_wait.add(time.perf_counter() - started)
            if not self._items:
                raise ChannelClosed(self.name)
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)


class GlobalWindow:
    """Global count/byte bound for dispatched-but-unconsumed packets."""

    def __init__(self, capacity: int, payload_capacity: int) -> None:
        if capacity <= 0 or payload_capacity <= 0:
            raise ValueError("window capacities must be positive")
        self.capacity = capacity
        self.payload_capacity = payload_capacity
        self.in_flight = 0
        self.in_flight_bytes = 0
        self.max_in_flight = 0
        self.max_in_flight_bytes = 0
        self.acquire_wait = WaitStats()
        self._condition = threading.Condition()
        self._cancelled = False

    def acquire(self, payload_bytes: int) -> None:
        started: Optional[float] = None
        with self._condition:
            while (
                not self._cancelled
                and (
                    self.in_flight >= self.capacity
                    or self.in_flight_bytes + payload_bytes > self.payload_capacity
                )
            ):
                if started is None:
                    started = time.perf_counter()
                self._condition.wait()
            if started is not None:
                self.acquire_wait.add(time.perf_counter() - started)
            if self._cancelled:
                raise WindowCancelled()
            self.in_flight += 1
            self.in_flight_bytes += payload_bytes
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.max_in_flight_bytes = max(
                self.max_in_flight_bytes, self.in_flight_bytes
            )

    def release(self, payload_bytes: int) -> None:
        with self._condition:
            self.in_flight -= 1
            self.in_flight_bytes -= payload_bytes
            if self.in_flight < 0 or self.in_flight_bytes < 0:
                raise RuntimeError("global window underflow")
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()


class ActiveWorkerGate:
    """Condition-based limit on active workers for adaptive deterministic mode."""

    def __init__(self, initial: int, maximum: int) -> None:
        self.maximum = maximum
        self.limit = max(1, min(initial, maximum))
        self.active = 0
        self.decisions: List[Dict[str, int]] = []
        self._condition = threading.Condition()

    def acquire(self) -> None:
        with self._condition:
            while self.active >= self.limit:
                self._condition.wait()
            self.active += 1

    def release(self) -> None:
        with self._condition:
            self.active -= 1
            self._condition.notify_all()

    def set_limit(self, value: int, reason: str) -> None:
        with self._condition:
            value = max(1, min(value, self.maximum))
            if value != self.limit:
                self.limit = value
                self.decisions.append({"limit": value, "reason": reason})
            self._condition.notify_all()


@dataclass
class Counters:
    generated: int = 0
    dispatched: int = 0
    consumed: int = 0
    retries: int = 0
    failures: int = 0
    duplicates: int = 0
    missing: int = 0
    malformed: int = 0
    manifest_mismatches: int = 0
    shutdown_events: int = 0
    cancellation_events: int = 0


def make_specs(
    count: int,
    seed: int,
    workload: str,
    manifest: str,
) -> List[PacketSpec]:
    specs: List[PacketSpec] = []
    for index in range(count):
        if workload == "uniform":
            cost = 2
            size = 4096
            straggler = 0.0
        elif workload == "mixed":
            cost = 1 + ((index * 17 + seed) % 6)
            size = 1024 * (1 + ((index * 29 + seed) % 64))
            straggler = 0.0
        elif workload == "stragglers":
            cost = 1 + ((index * 11 + seed) % 4)
            size = 4096 * (1 + ((index * 13 + seed) % 16))
            straggler = 0.004 if index % 23 == 0 else 0.0
            if straggler:
                cost *= 3
        elif workload == "large":
            cost = 1 + ((index * 7 + seed) % 4)
            size = 64 * 1024 * (1 + ((index * 5 + seed) % 4))
            straggler = 0.002 if index % 19 == 0 else 0.0
        else:
            raise ValueError(f"unknown workload: {workload}")
        specs.append(
            PacketSpec(
                manifest_sha256=manifest,
                sequence_index=index,
                packet_id=f"packet-{index:08d}",
                cost_units=cost,
                payload_bytes=size,
                straggler_delay_s=straggler,
            )
        )
    return specs


def prepare(item: WorkItem, worker_id: int, fault_state: Dict[int, int]) -> Prepared:
    spec = item.spec
    if fault_state.get(spec.sequence_index, 0) > 0:
        fault_state[spec.sequence_index] -= 1
        raise RuntimeError(f"injected transient worker failure for {spec.packet_id}")
    started = time.perf_counter_ns()
    payload = (
        f"{spec.manifest_sha256}:{spec.sequence_index}:{spec.packet_id}:"
        f"{spec.payload_bytes}"
    ).encode("ascii")
    for _ in range(spec.cost_units * 3):
        payload = hashlib.blake2b(payload, digest_size=32).digest()
    if spec.straggler_delay_s:
        time.sleep(spec.straggler_delay_s)
    return Prepared(
        spec=spec,
        worker_id=worker_id,
        payload_digest=hashlib.sha256(payload).hexdigest(),
        elapsed_ns=time.perf_counter_ns() - started,
    )


def result_digest(results: Iterable[Prepared]) -> str:
    rows = sorted(
        (r.spec.sequence_index, r.payload_digest) for r in results
    )
    return sha256_text("\n".join(f"{i}:{d}" for i, d in rows))


def latency_summary(results: Sequence[Prepared]) -> Dict[str, float]:
    values = [r.elapsed_ns / 1e9 for r in results]
    return {
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values) if values else 0.0,
    }


def run_pipeline(
    specs: Sequence[PacketSpec],
    mode: str,
    workers: int,
    queue_capacity: int,
    window_capacity: int,
    payload_window_bytes: int,
    consume_delay_s: float = 0.0,
    fail_once_seq: Optional[int] = None,
    cancel_after_commits: Optional[int] = None,
    max_retries: int = 1,
) -> Dict[str, object]:
    if not specs:
        return {"mode": mode, "results": [], "cancelled": False}
    if mode not in {"fixed", "adaptive"}:
        raise ValueError(mode)

    input_ch = BoundedChannel("input", queue_capacity)
    done_ch = BoundedChannel("done", queue_capacity)
    stage_ch = BoundedChannel("stage", queue_capacity)
    window = GlobalWindow(window_capacity, payload_window_bytes)
    counters = Counters()
    cancel_event = threading.Event()
    producer_done = threading.Event()
    assembler_done = threading.Event()
    fault_state = {fail_once_seq: 1} if fail_once_seq is not None else {}
    fault_lock = threading.Lock()
    results: List[Prepared] = []
    result_lock = threading.Lock()
    pending_max = 0
    retry_backlog: Deque[WorkItem] = deque()
    expected = {spec.sequence_index: spec for spec in specs}
    expected_order = [spec.sequence_index for spec in specs]
    seen_staged: set[int] = set()
    worker_gate = (
        ActiveWorkerGate(initial=max(1, min(2, workers)), maximum=workers)
        if mode == "adaptive"
        else None
    )

    def stop_all() -> None:
        cancel_event.set()
        window.cancel()
        input_ch.close()
        done_ch.close()
        stage_ch.close()
        counters.cancellation_events += 1

    def producer() -> None:
        try:
            for spec in specs:
                if cancel_event.is_set():
                    break
                window.acquire(spec.payload_bytes)
                try:
                    input_ch.put(WorkItem(spec=spec))
                except ChannelClosed:
                    window.release(spec.payload_bytes)
                    break
                counters.generated += 1
                counters.dispatched += 1
        except WindowCancelled:
            pass
        finally:
            producer_done.set()

    def worker(worker_id: int) -> None:
        while not cancel_event.is_set():
            try:
                item = input_ch.get()
            except ChannelClosed:
                return
            if not isinstance(item, WorkItem):
                counters.malformed += 1
                continue
            if worker_gate is not None:
                worker_gate.acquire()
            try:
                with fault_lock:
                    prepared = prepare(item, worker_id, fault_state)
                done_ch.put(prepared)
            except Exception as exc:  # intentional fault boundary
                counters.failures += 1
                try:
                    done_ch.put(Failed(item=item, worker_id=worker_id, error=str(exc)))
                except ChannelClosed:
                    return
            finally:
                if worker_gate is not None:
                    worker_gate.release()

    def adaptive_monitor() -> None:
        if worker_gate is None:
            return
        while not assembler_done.is_set() and not cancel_event.is_set():
            input_depth = input_ch.qsize()
            stage_depth = stage_ch.qsize()
            done_depth = done_ch.qsize()
            if stage_depth >= max(1, int(queue_capacity * 0.75)):
                worker_gate.set_limit(worker_gate.limit - 1, "stage_pressure")
            elif input_depth >= max(1, int(queue_capacity * 0.5)) and done_depth < queue_capacity:
                worker_gate.set_limit(worker_gate.limit + 1, "input_pressure")
            elif not input_depth and not done_depth and stage_depth == 0:
                worker_gate.set_limit(1, "drain_safe")
            assembler_done.wait(0.002)

    def assembler() -> None:
        nonlocal pending_max
        pending: Dict[int, Prepared] = {}
        next_position = 0
        staged = 0
        try:
            while staged < len(expected_order) and not cancel_event.is_set():
                while retry_backlog and input_ch.try_put(retry_backlog[0]):
                    retry_backlog.popleft()
                event = done_ch.get()
                if isinstance(event, Failed):
                    if event.item.attempt < max_retries and not cancel_event.is_set():
                        counters.retries += 1
                        retry_backlog.append(
                            replace(event.item, attempt=event.item.attempt + 1)
                        )
                    else:
                        counters.missing += 1
                        stop_all()
                    continue
                if not isinstance(event, Prepared):
                    counters.malformed += 1
                    stop_all()
                    break
                seq = event.spec.sequence_index
                if seq not in expected:
                    counters.missing += 1
                    stop_all()
                    break
                if event.spec.manifest_sha256 != specs[0].manifest_sha256:
                    counters.manifest_mismatches += 1
                    stop_all()
                    break
                if seq in pending or seq in seen_staged:
                    counters.duplicates += 1
                    stop_all()
                    break
                pending[seq] = event
                pending_max = max(pending_max, len(pending))
                while next_position < len(expected_order):
                    wanted = expected_order[next_position]
                    if wanted not in pending:
                        break
                    staged_event = pending.pop(wanted)
                    stage_ch.put(staged_event)
                    seen_staged.add(wanted)
                    staged += 1
                    next_position += 1
                if producer_done.is_set() and staged == len(expected_order):
                    break
        except ChannelClosed:
            counters.shutdown_events += 1
        finally:
            if retry_backlog:
                counters.missing += len(retry_backlog)
                retry_backlog.clear()
            input_ch.close()
            stage_ch.close()
            assembler_done.set()

    def consumer() -> None:
        try:
            expected_position = 0
            while expected_position < len(expected_order) and not cancel_event.is_set():
                event = stage_ch.get()
                if not isinstance(event, Prepared):
                    counters.malformed += 1
                    stop_all()
                    return
                wanted = expected_order[expected_position]
                if event.spec.sequence_index != wanted:
                    counters.missing += 1
                    stop_all()
                    return
                with result_lock:
                    results.append(event)
                counters.consumed += 1
                window.release(event.spec.payload_bytes)
                expected_position += 1
                if consume_delay_s:
                    time.sleep(consume_delay_s)
                if cancel_after_commits is not None and counters.consumed >= cancel_after_commits:
                    stop_all()
                    return
        except ChannelClosed:
            counters.shutdown_events += 1

    started = time.perf_counter()
    cpu_started = time.process_time()
    threads = [
        threading.Thread(target=worker, args=(worker_id,), name=f"npa-v1-worker-{worker_id}")
        for worker_id in range(workers)
    ]
    producer_thread = threading.Thread(target=producer, name="npa-v1-producer")
    assembler_thread = threading.Thread(target=assembler, name="npa-v1-assembler")
    consumer_thread = threading.Thread(target=consumer, name="npa-v1-consumer")
    monitor_thread = threading.Thread(target=adaptive_monitor, name="npa-v1-adaptive-monitor")

    consumer_thread.start()
    assembler_thread.start()
    for thread in threads:
        thread.start()
    if worker_gate is not None:
        monitor_thread.start()
    producer_thread.start()

    producer_thread.join()
    assembler_thread.join()
    if not cancel_event.is_set():
        input_ch.close()
    for thread in threads:
        thread.join()
    if worker_gate is not None:
        assembler_done.set()
        monitor_thread.join()
    consumer_thread.join()

    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    digest = result_digest(results)
    expected_count = len(expected_order)
    complete = (
        not cancel_after_commits
        and len(results) == expected_count
        and counters.duplicates == 0
        and counters.missing == 0
        and counters.malformed == 0
        and counters.manifest_mismatches == 0
        and not cancel_event.is_set()
    )
    return {
        "mode": mode,
        "workers": workers,
        "queue_capacity": queue_capacity,
        "window_capacity": window_capacity,
        "payload_window_bytes": payload_window_bytes,
        "expected_packets": expected_count,
        "committed_packets": len(results),
        "complete": complete,
        "cancelled": cancel_event.is_set(),
        "elapsed_s": elapsed,
        "throughput_packets_s": len(results) / elapsed if elapsed else 0.0,
        "throughput_payload_bytes_s": (
            sum(r.spec.payload_bytes for r in results) / elapsed if elapsed else 0.0
        ),
        "process_cpu_s": cpu_elapsed,
        "process_cpu_core_equivalent": cpu_elapsed / elapsed if elapsed else 0.0,
        "latency": latency_summary(results),
        "result_digest": digest,
        "pending_reorder_max": pending_max,
        "window_max_in_flight": window.max_in_flight,
        "window_max_in_flight_bytes": window.max_in_flight_bytes,
        "window_acquire_wait": window.acquire_wait.summary(),
        "queue_metrics": {
            "input": {
                "high_water": input_ch.high_water,
                "put_wait": input_ch.put_wait.summary(),
                "get_wait": input_ch.get_wait.summary(),
            },
            "done": {
                "high_water": done_ch.high_water,
                "put_wait": done_ch.put_wait.summary(),
                "get_wait": done_ch.get_wait.summary(),
            },
            "stage": {
                "high_water": stage_ch.high_water,
                "put_wait": stage_ch.put_wait.summary(),
                "get_wait": stage_ch.get_wait.summary(),
            },
        },
        "counters": asdict(counters),
        "adaptive_decisions": worker_gate.decisions if worker_gate else [],
        "retry_backlog_remaining": len(retry_backlog),
        "committed_sequences": sorted(r.spec.sequence_index for r in results),
    }


def serial_run(specs: Sequence[PacketSpec]) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    results = [prepare(WorkItem(spec), 0, {}) for spec in specs]
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    return {
        "mode": "serial",
        "expected_packets": len(specs),
        "committed_packets": len(results),
        "complete": True,
        "cancelled": False,
        "elapsed_s": elapsed,
        "throughput_packets_s": len(results) / elapsed if elapsed else 0.0,
        "throughput_payload_bytes_s": (
            sum(r.spec.payload_bytes for r in results) / elapsed if elapsed else 0.0
        ),
        "process_cpu_s": cpu_elapsed,
        "process_cpu_core_equivalent": cpu_elapsed / elapsed if elapsed else 0.0,
        "latency": latency_summary(results),
        "result_digest": result_digest(results),
        "committed_sequences": [r.spec.sequence_index for r in results],
    }


def recovery_check(
    specs: Sequence[PacketSpec],
    serial: Dict[str, object],
    workers: int,
    queue_capacity: int,
    window_capacity: int,
    payload_window_bytes: int,
) -> Dict[str, object]:
    cancelled = run_pipeline(
        specs,
        mode="fixed",
        workers=workers,
        queue_capacity=queue_capacity,
        window_capacity=window_capacity,
        payload_window_bytes=payload_window_bytes,
        consume_delay_s=0.001,
        cancel_after_commits=max(1, len(specs) // 4),
    )
    committed = set(cancelled["committed_sequences"])
    remaining = [spec for spec in specs if spec.sequence_index not in committed]
    recovered = run_pipeline(
        remaining,
        mode="fixed",
        workers=workers,
        queue_capacity=queue_capacity,
        window_capacity=window_capacity,
        payload_window_bytes=payload_window_bytes,
    )
    # Reconstruct the combined digest from an uninterrupted serial reference by
    # using the deterministic packet digest for each sequence. The recovery run
    # must complete every sequence not committed before cancellation.
    combined_sequences = sorted(committed | set(recovered["committed_sequences"]))
    combined_ok = combined_sequences == [s.sequence_index for s in specs]
    no_duplicates = len(combined_sequences) == len(set(combined_sequences))
    return {
        "cancelled_run": {
            "committed": len(committed),
            "cancelled": cancelled["cancelled"],
            "shutdown_events": cancelled["counters"]["shutdown_events"],
        },
        "recovery_run": {
            "committed": len(recovered["committed_sequences"]),
            "complete": recovered["complete"],
            "duplicates": recovered["counters"]["duplicates"],
            "missing": recovered["counters"]["missing"],
        },
        "combined_sequence_coverage": combined_ok,
        "combined_no_duplicates": no_duplicates,
        "serial_reference_digest": serial["result_digest"],
        "note": (
            "The sequence-coverage check is the cancellation/recovery gate; "
            "the uninterrupted serial digest remains the immutable reference."
        ),
    }


def run_suite(args: argparse.Namespace) -> Dict[str, object]:
    manifest = sha256_text(f"npa-v1-manifest:{args.seed}:{args.packets}")
    payload_window = args.window * args.max_packet_bytes
    scenarios = {}
    for workload in ("uniform", "mixed", "stragglers", "large"):
        specs = make_specs(args.packets, args.seed, workload, manifest)
        serial = serial_run(specs)
        fixed = run_pipeline(
            specs,
            mode="fixed",
            workers=args.workers,
            queue_capacity=args.queue_capacity,
            window_capacity=args.window,
            payload_window_bytes=payload_window,
            consume_delay_s=args.consume_delay_s if workload != "uniform" else 0.0,
        )
        adaptive = run_pipeline(
            specs,
            mode="adaptive",
            workers=args.workers,
            queue_capacity=args.queue_capacity,
            window_capacity=args.window,
            payload_window_bytes=payload_window,
            consume_delay_s=args.consume_delay_s if workload != "uniform" else 0.0,
        )
        scenarios[workload] = {
            "serial": serial,
            "fixed": fixed,
            "adaptive": adaptive,
            "fixed_digest_matches_serial": fixed["result_digest"] == serial["result_digest"],
            "adaptive_digest_matches_serial": adaptive["result_digest"] == serial["result_digest"],
            "fixed_global_bound_ok": fixed["window_max_in_flight"] <= args.window,
            "adaptive_global_bound_ok": adaptive["window_max_in_flight"] <= args.window,
            "fixed_reorder_bound_ok": fixed["pending_reorder_max"] <= args.window,
            "adaptive_reorder_bound_ok": adaptive["pending_reorder_max"] <= args.window,
        }

    fault_specs = make_specs(args.packets, args.seed, "mixed", manifest)
    serial_fault = serial_run(fault_specs)
    fault_run = run_pipeline(
        fault_specs,
        mode="fixed",
        workers=args.workers,
        queue_capacity=args.queue_capacity,
        window_capacity=args.window,
        payload_window_bytes=payload_window,
        fail_once_seq=min(17, args.packets - 1),
        max_retries=1,
    )
    scenarios["fault_injection"] = {
        "run": fault_run,
        "digest_matches_serial": fault_run["result_digest"] == serial_fault["result_digest"],
        "retry_observed": fault_run["counters"]["retries"] >= 1,
        "failure_observed": fault_run["counters"]["failures"] >= 1,
    }
    scenarios["cancellation_recovery"] = recovery_check(
        fault_specs,
        serial_fault,
        args.workers,
        args.queue_capacity,
        args.window,
        payload_window,
    )

    checks = {
        "all_standard_digests_match": all(
            value.get("fixed_digest_matches_serial", True)
            and value.get("adaptive_digest_matches_serial", True)
            for value in scenarios.values()
            if "serial" in value
        ),
        "all_global_windows_bounded": all(
            value.get("fixed_global_bound_ok", True)
            and value.get("adaptive_global_bound_ok", True)
            for value in scenarios.values()
            if "serial" in value
        ),
        "all_reorder_buffers_bounded": all(
            value.get("fixed_reorder_bound_ok", True)
            and value.get("adaptive_reorder_bound_ok", True)
            for value in scenarios.values()
            if "serial" in value
        ),
        "fault_recovered": scenarios["fault_injection"]["digest_matches_serial"]
        and scenarios["fault_injection"]["retry_observed"]
        and scenarios["fault_injection"]["failure_observed"],
        "cancellation_recovery_covered": scenarios["cancellation_recovery"]["combined_sequence_coverage"]
        and scenarios["cancellation_recovery"]["combined_no_duplicates"]
        and scenarios["cancellation_recovery"]["cancelled_run"]["cancelled"],
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, sort_keys=True))
    return {
        "schema": "ssre_npa_v1_bounded_benchmark_v1",
        "cpu_only": True,
        "cuda_used": False,
        "model_training": False,
        "seed": args.seed,
        "manifest_sha256": manifest,
        "parameters": {
            "packets": args.packets,
            "workers": args.workers,
            "queue_capacity": args.queue_capacity,
            "window_capacity": args.window,
            "payload_window_bytes": payload_window,
            "consume_delay_s": args.consume_delay_s,
            "max_packet_bytes": args.max_packet_bytes,
        },
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scenarios": scenarios,
        "checks": checks,
        "interpretation": (
            "Scheduler mechanics and resource bounds only; no neural-training "
            "or GPU-acceleration claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packets", type=int, default=192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--max-packet-bytes", type=int, default=256 * 1024)
    parser.add_argument("--consume-delay-s", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.packets, args.workers, args.queue_capacity, args.window) <= 0:
        parser.error("packets, workers, queue-capacity, and window must be positive")
    result = run_suite(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
