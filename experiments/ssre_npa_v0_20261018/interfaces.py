"""Small, transport-agnostic SSRE-NPA v0 contracts.

This module is intentionally independent of Crucible, PyTorch, CUDA, and the
historical SSRE runtime.  It defines data contracts only; the CPU-only
synthetic benchmark exercises the ordering/integrity portions of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class WorkPacket:
    """Immutable scheduled unit of preparation work."""

    manifest_sha256: str
    sequence_index: int
    packet_id: str
    sample_ids: Tuple[str, ...]
    payload_ref: str
    target_bytes: int
    estimated_bytes: int
    integrity_sha256: str


@dataclass(frozen=True)
class PacketResult:
    """Prepared result that must be reassembled by sequence_index."""

    manifest_sha256: str
    sequence_index: int
    packet_id: str
    worker_id: int
    payload_digest: str
    prepared_bytes: int
    elapsed_ns: int
    integrity_ok: bool


@dataclass(frozen=True)
class ResourceConfig:
    """Prevalidated scheduler operating point."""

    name: str
    worker_count: int
    input_queue_capacity: int
    completion_queue_capacity: int
    staging_queue_capacity: int
    prefetch_window: int
    pinned_memory_bytes: int
    reserved_cpu_cores: Tuple[int, ...] = ()


@dataclass(frozen=True)
class TelemetrySample:
    """One scheduler interval; GPU fields may be absent in CPU-only tests."""

    interval_index: int
    queue_depths: Tuple[Tuple[str, int], ...]
    queue_high_watermarks: Tuple[Tuple[str, int], ...]
    blocked_puts: int
    worker_completed: int
    reorder_buffer_size: int
    cpu_utilization: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_wait_fraction: Optional[float] = None
    host_memory_bytes: Optional[int] = None
    vram_bytes: Optional[int] = None


@dataclass(frozen=True)
class SchedulerDecision:
    """Auditable controller decision."""

    interval_index: int
    controller_name: str
    controller_version: str
    config_name: str
    confidence: float
    input_telemetry_sha256: str
    fallback_reason: Optional[str]
    decision_latency_ns: int


@dataclass
class IntegrityCounters:
    """Mutable counters used by a run, serialized only after completion."""

    packets_generated: int = 0
    packets_consumed: int = 0
    duplicate_packets: int = 0
    missing_packets: int = 0
    malformed_packets: int = 0
    manifest_mismatches: int = 0
    blocked_puts: int = 0
    max_reorder_buffer: int = 0
    extra: dict = field(default_factory=dict)
