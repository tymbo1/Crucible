# SSRE-NPA v0

## Standalone neural pipeline accelerator specification

**Status:** design and CPU-only synthetic validation artifact  
**Date:** 2026-10-18  
**Scope:** host-side preparation and scheduling around a frozen neural training workload

SSRE-NPA means **SSRE Neural Pipeline Accelerator**. It is deliberately separate from the historical SSRE/WSE implementation and from Crucible's training code. It may reuse scheduling ideas that were recovered from SSRE, but it does not assume that the unavailable `ssre_runtime_v4_1` source exists.

The first objective is not to make the CPU perform neural matrix multiplication. It is to keep a GPU supplied with deterministic, valid work while avoiding CPU saturation, queue explosions, ordering changes, and hidden synchronization.

## Non-negotiable invariants

1. The scientific workload is immutable: sample order, packet IDs, batch boundaries, masks, targets, seeds, optimizer settings, and model parameters are unchanged.
2. Worker completion order may vary, but consumption order is the frozen schedule order.
3. Every packet carries a manifest identifier and provenance references.
4. Queues are bounded. Backpressure is explicit and measured.
5. Malformed packets, missing data, provenance mismatches, and duplicate sequence IDs fail closed.
6. The neural controller may choose only from prevalidated resource configurations. It cannot choose samples or alter training semantics.
7. The deterministic heuristic scheduler is always available as a fallback.
8. No controller decision is accepted without an audit record containing the input telemetry, selected configuration, and reason.
9. CPU scheduling experiments must reserve resources for the GPU driver/coordinator path.
10. This package must not modify or restart any Crucible run.

## Data path

```text
immutable manifest/schedule
        |
        v
deterministic packet generator
        |
        v
bounded CPU preparation reservoir
  persistent workers, local caches,
  validation, token/batch assembly
        |
        v
ordered completion/reassembly
        |
        v
bounded pinned-memory staging queue
        |
        v
nonblocking H2D transfer
        |
        v
GPU forward/backward/optimizer
```

The current v0 synthetic benchmark stops before H2D/GPU execution. It tests the parts that can be validated without touching the active experiment: deterministic packet creation, bounded queues, out-of-order worker completion, ordered reassembly, and fail-closed integrity checks.

## Controllers

### Fast deterministic controller

This controller owns packet execution. It uses persistent workers, bounded queues, deterministic packet IDs, and ordered assembly. Scheduling may change which worker handles a packet, but never which packet is consumed next.

Candidate controls:

- worker count;
- input queue depth;
- completion queue depth;
- staging queue depth;
- prefetch window;
- pinned-memory budget;
- CPU affinity/reserved coordinator cores;
- whether a safe local cache is enabled.

### Slow neural controller

The neural controller is a future optional module. It runs every fixed decision interval, not every packet. Its output is an index into a finite table of safe `ResourceConfig` values.

It may estimate queue depletion, preparation time, GPU wait probability, CPU contention, and memory pressure. It must not alter data order or model semantics. If it is unavailable, slow, non-finite, or outside a declared confidence envelope, the deterministic heuristic controller remains active.

Suggested v0 model: a CPU-resident 32-input, 32-hidden-unit GRU with decisions every 64 training steps. This is a design target, not implemented or trained here.

## Proposed telemetry

At each packet/update and at each controller decision interval, record:

- packet sequence index and manifest ID;
- preparation start/end and duration;
- queue depth and high-water mark for each bounded queue;
- number and duration of blocked puts;
- worker utilization and per-worker completion counts;
- ordered-assembly lag and reorder-buffer size;
- staging wait time;
- host-to-device copy time;
- GPU active time and GPU wait-for-work time;
- CPU utilization, reserved-core utilization, and memory pressure;
- peak host RAM and VRAM;
- selected `ResourceConfig` and fallback reason;
- packet/result digest and integrity status.

The active scientific run's telemetry must remain in its own frozen artifacts. This package does not attach to that run.

## Reward and safety model

For offline controller training only, use a cost-aware objective such as:

```text
R = valid_tokens_per_second
    - lambda_gpu * gpu_wait_fraction
    - mu_cpu * coordinator_contention
    - nu_mem * memory_pressure
    - rho_ctrl * controller_overhead
```

Integrity violations are hard failures, not soft reward penalties. A controller that improves throughput by changing order, dropping packets, changing masks, or starving the GPU coordinator is invalid.

The learned controller must beat the deterministic heuristic scheduler after including its own inference and decision overhead. A neural controller that merely beats the unoptimized reference is not sufficient.

## Isolation boundary

This directory is a standalone design/profiling artifact. It does not import Crucible, Braidformer, PyTorch, CUDA, RunPod credentials, or the historical SSRE runtime. A later integration should be a separate reviewed adapter with its own manifest and tests.
