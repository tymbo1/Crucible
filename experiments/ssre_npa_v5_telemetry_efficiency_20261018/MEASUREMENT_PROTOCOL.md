# SSRE-NPA v5 measurement protocol

## Frozen workload

- Context-128 E16 Branch-B Crucible package.
- Seed 42 anchor and schedule prefix.
- Same source indices, masks, batch boundaries, model geometry, optimizer and
  learning-rate contract for every backend.
- Two warm-up optimizer steps and eight timed optimizer steps.
- Two repetitions per backend.
- CUDA BF16 with deterministic algorithm enforcement in warning mode because
  the frozen telemetry path uses a CUDA `bincount` without a deterministic
  implementation.

## Telemetry invariants

The full optimized backends preserve every field emitted by the frozen class:

- routed-token totals;
- hard/top-1/top-2 expert counts;
- expert-by-byte affinity;
- selected-token counts;
- selected routing-probability sums/counts;
- entropy;
- expert-output activation norm sums/counts;
- derived CV, max:min load ratio and normalized entropy.

The minimal backend preserves the primary evaluator fields: routed-token
totals, hard/top-1/top-2 counts, expert-by-byte affinity, entropy and derived
load metrics. Optional selected-token and activation-norm diagnostics are
intentionally omitted and are not claimed equivalent by that backend.

Integer counters must match exactly. Floating telemetry fields use an
explicit absolute tolerance of `1e-6`, after matching the frozen float32
dtype for derived load scalars. Model-state hashes and loss trajectories use
the v4 `1e-5` loss tolerance and must match exactly under the deterministic
control.

## Timing

The benchmark measures synchronized CPU preparation, H2D transfer, forward,
backward, optimizer/clip and non-compute proxy time. Summary extraction and
JSON serialization are measured separately after the timed optimizer window,
matching the frozen runner's checkpoint-reporting boundary.

The optimized full backend packs all layer/field tensors into one device
buffer and performs one host transfer. The buffered backend defers report
serialization; filesystem fsync is not placed in the GPU critical path by
this benchmark and is reported as unmeasured.

## Profiler probe

One separate disposable batch is profiled with forward attention and MoE
module labels, CUDA operator aggregation for routing/indexing operations, and
synchronized CUDA events for backward and optimizer time. The probe is
descriptive and does not alter the benchmark measurements.

