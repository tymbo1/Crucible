# SSRE-NPA v0 interfaces

The reference Python interfaces are in `interfaces.py`. They are intentionally small and transport-agnostic so the first implementation can use threads, processes, shared memory, or a future native queue without changing the scientific contract.

## `WorkPacket`

Immutable unit of preparation work:

- `manifest_sha256` binds the packet to the frozen dataset contract;
- `sequence_index` is the only ordering authority;
- `sample_ids` identify the exact examples;
- `payload_ref` points to precomputed data or a deterministic source record;
- `target_bytes` and `estimated_bytes` support resource accounting;
- `integrity_sha256` identifies the packet contents.

## `PacketResult`

Worker output:

- repeats the packet/manifest identity;
- contains the prepared batch reference or payload digest;
- records worker ID and preparation timing;
- carries an integrity status;
- is not consumable until the ordered assembler has validated it.

## `ResourceConfig`

One prevalidated operating point. It includes worker count, queue depths, staging depth, pinned-memory budget, and reserved CPU cores. A controller chooses an existing configuration; it does not synthesize arbitrary values during a run.

## `TelemetrySample`

One interval of observed queue, worker, CPU, memory, staging, and GPU state. The v0 synthetic benchmark emits a reduced CPU-only subset.

## `SchedulerDecision`

Auditable controller output: interval, input telemetry digest, selected configuration, controller name/version, confidence, fallback reason, and decision latency.

## Future adapters

A Crucible adapter must provide:

1. a deterministic packet iterator derived from the frozen schedule;
2. a packet validator that checks manifest and provenance binding;
3. an ordered batch sink;
4. measurements for H2D/GPU wait time;
5. a clean shutdown and resume contract.

No adapter is implemented in v0.
