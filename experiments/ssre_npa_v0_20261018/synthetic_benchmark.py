#!/usr/bin/env python3
"""CPU-only mechanical benchmark for the SSRE-NPA v0 pipeline contract.

It compares a serial reference with a bounded worker reservoir.  The reservoir
deliberately allows workers to finish out of order, then reassembles results by
the immutable sequence index before consumption.  It does not import torch,
CUDA, Crucible, or any historical SSRE module.

This is not a neural-training benchmark.  Its purpose is to validate ordering,
bounded queues, backpressure accounting, deterministic packet content, and
fail-closed integrity checks before a real adapter is designed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from interfaces import IntegrityCounters, PacketResult, WorkPacket


SENTINEL = object()


class BoundedPutStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.blocked_puts = 0

    def put(self, q: queue.Queue, item: object) -> None:
        while True:
            try:
                q.put(item, timeout=0.001)
                return
            except queue.Full:
                with self._lock:
                    self.blocked_puts += 1


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_packet(index: int, manifest: str, work_scale: int) -> WorkPacket:
    sample_id = f"synthetic-{index:08d}"
    payload = f"{manifest}:{index}:{work_scale}".encode("utf-8")
    packet_digest = digest_bytes(payload)
    estimated = 256 + ((index * 37) % 513)
    return WorkPacket(
        manifest_sha256=manifest,
        sequence_index=index,
        packet_id=f"packet-{index:08d}",
        sample_ids=(sample_id,),
        payload_ref=f"synthetic://{sample_id}",
        target_bytes=estimated,
        estimated_bytes=estimated,
        integrity_sha256=packet_digest,
    )


def packet_payload(packet: WorkPacket, work_scale: int) -> bytes:
    base = (
        f"{packet.manifest_sha256}:{packet.sequence_index}:"
        f"{packet.packet_id}:{work_scale}"
    ).encode("utf-8")
    data = base
    for _ in range(max(1, work_scale)):
        data = hashlib.sha256(data).digest() + base
    return data


def process_packet(packet: WorkPacket, worker_id: int, work_scale: int) -> PacketResult:
    started = time.perf_counter_ns()
    payload = packet_payload(packet, work_scale)
    # Keep work deterministic but make completion order nonuniform.
    for _ in range((packet.sequence_index * 13) % 5):
        payload = hashlib.sha256(payload).digest() + payload[:16]
    elapsed = time.perf_counter_ns() - started
    return PacketResult(
        manifest_sha256=packet.manifest_sha256,
        sequence_index=packet.sequence_index,
        packet_id=packet.packet_id,
        worker_id=worker_id,
        payload_digest=digest_bytes(payload),
        prepared_bytes=len(payload),
        elapsed_ns=elapsed,
        integrity_ok=True,
    )


def serial_run(
    packets: List[WorkPacket],
    work_scale: int,
) -> Tuple[List[PacketResult], Dict[str, object]]:
    started = time.perf_counter()
    results = [process_packet(packet, 0, work_scale) for packet in packets]
    elapsed = time.perf_counter() - started
    return results, {
        "mode": "serial_reference",
        "elapsed_s": elapsed,
        "packets": len(results),
        "result_order_ok": [r.sequence_index for r in results]
        == list(range(len(packets))),
    }


def reservoir_run(
    packets: List[WorkPacket],
    worker_count: int,
    queue_capacity: int,
    work_scale: int,
    consume_delay_s: float,
) -> Tuple[List[PacketResult], Dict[str, object]]:
    started = time.perf_counter()
    input_q: queue.Queue = queue.Queue(maxsize=queue_capacity)
    done_q: queue.Queue = queue.Queue(maxsize=queue_capacity)
    stage_q: queue.Queue = queue.Queue(maxsize=queue_capacity)
    input_puts = BoundedPutStats()
    done_puts = BoundedPutStats()
    stage_puts = BoundedPutStats()
    counters = IntegrityCounters()
    workers_done = threading.Event()
    consumed: List[PacketResult] = []
    consumed_lock = threading.Lock()
    high_water = {"input": 0, "done": 0, "stage": 0}
    high_water_lock = threading.Lock()

    def update_high_water(name: str, q: queue.Queue) -> None:
        with high_water_lock:
            high_water[name] = max(high_water[name], q.qsize())

    def producer() -> None:
        for packet in packets:
            input_puts.put(input_q, packet)
            counters.packets_generated += 1
            update_high_water("input", input_q)
        for _ in range(worker_count):
            input_puts.put(input_q, SENTINEL)
            update_high_water("input", input_q)

    def worker(worker_id: int) -> None:
        while True:
            item = input_q.get()
            if item is SENTINEL:
                return
            if not isinstance(item, WorkPacket):
                counters.malformed_packets += 1
                continue
            result = process_packet(item, worker_id, work_scale)
            done_puts.put(done_q, result)
            update_high_water("done", done_q)

    def assembler() -> None:
        next_index = 0
        pending: Dict[int, PacketResult] = {}
        while next_index < len(packets):
            result = done_q.get()
            if not isinstance(result, PacketResult) or not result.integrity_ok:
                counters.malformed_packets += 1
                continue
            if result.manifest_sha256 != packets[0].manifest_sha256:
                counters.manifest_mismatches += 1
                continue
            if result.sequence_index in pending or result.sequence_index < next_index:
                counters.duplicate_packets += 1
                continue
            pending[result.sequence_index] = result
            counters.max_reorder_buffer = max(
                counters.max_reorder_buffer, len(pending)
            )
            while next_index in pending:
                stage_puts.put(stage_q, pending.pop(next_index))
                update_high_water("stage", stage_q)
                next_index += 1
        workers_done.set()

    def consumer() -> None:
        expected = 0
        while expected < len(packets):
            result = stage_q.get()
            if result.sequence_index != expected:
                counters.missing_packets += 1
                raise RuntimeError(
                    f"ordered consumer expected {expected}, "
                    f"got {result.sequence_index}"
                )
            with consumed_lock:
                consumed.append(result)
            counters.packets_consumed += 1
            expected += 1
            if consume_delay_s:
                time.sleep(consume_delay_s)

    producer_thread = threading.Thread(target=producer, name="npa-producer")
    worker_threads = [
        threading.Thread(target=worker, args=(i,), name=f"npa-worker-{i}")
        for i in range(worker_count)
    ]
    assembler_thread = threading.Thread(target=assembler, name="npa-assembler")
    consumer_thread = threading.Thread(target=consumer, name="npa-consumer")

    consumer_thread.start()
    assembler_thread.start()
    for thread in worker_threads:
        thread.start()
    producer_thread.start()

    producer_thread.join()
    for thread in worker_threads:
        thread.join()
    assembler_thread.join()
    consumer_thread.join()

    elapsed = time.perf_counter() - started
    counters.blocked_puts = (
        input_puts.blocked_puts
        + done_puts.blocked_puts
        + stage_puts.blocked_puts
    )
    result_order = [result.sequence_index for result in consumed]
    result_digest = digest_bytes(
        "\n".join(
            f"{r.sequence_index}:{r.payload_digest}" for r in consumed
        ).encode("ascii")
    )
    return consumed, {
        "mode": "bounded_reservoir",
        "elapsed_s": elapsed,
        "packets": len(consumed),
        "workers": worker_count,
        "queue_capacity": queue_capacity,
        "consume_delay_s": consume_delay_s,
        "result_order_ok": result_order == list(range(len(packets))),
        "result_digest": result_digest,
        "queue_high_water": high_water,
        "integrity": asdict(counters),
        "workers_finished": workers_done.is_set(),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    manifest = digest_bytes(
        f"synthetic-manifest:{args.seed}:{args.packets}".encode("ascii")
    )
    packets = [
        make_packet(index, manifest, args.work_scale)
        for index in range(args.packets)
    ]
    serial_results, serial_metrics = serial_run(packets, args.work_scale)
    reservoir_results, reservoir_metrics = reservoir_run(
        packets,
        args.workers,
        args.queue_capacity,
        args.work_scale,
        args.consume_delay_s,
    )
    serial_digest = digest_bytes(
        "\n".join(
            f"{r.sequence_index}:{r.payload_digest}" for r in serial_results
        ).encode("ascii")
    )
    result = {
        "schema": "ssre_npa_synthetic_benchmark_v1",
        "cpu_only": True,
        "model_training": False,
        "cuda_used": False,
        "seed": args.seed,
        "manifest_sha256": manifest,
        "parameters": {
            "packets": args.packets,
            "workers": args.workers,
            "queue_capacity": args.queue_capacity,
            "work_scale": args.work_scale,
            "consume_delay_s": args.consume_delay_s,
        },
        "serial": {**serial_metrics, "result_digest": serial_digest},
        "reservoir": reservoir_metrics,
        "equivalence": {
            "ordered_digest_matches_serial": (
                serial_digest == reservoir_metrics["result_digest"]
            ),
            "all_packets_consumed": (
                reservoir_metrics["integrity"]["packets_consumed"]
                == args.packets
            ),
            "bounded_queues_respected": all(
                value <= args.queue_capacity
                for value in reservoir_metrics["queue_high_water"].values()
            ),
        },
        "interpretation": (
            "Mechanical pipeline validation only; throughput is not a "
            "neural-training or GPU-acceleration claim."
        ),
    }
    if not all(result["equivalence"].values()):
        raise RuntimeError(json.dumps(result["equivalence"], sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packets", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--work-scale", type=int, default=4)
    parser.add_argument("--consume-delay-s", type=float, default=0.0002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.packets <= 0 or args.workers <= 0 or args.queue_capacity <= 0:
        parser.error("packets, workers, and queue-capacity must be positive")
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
