# SSRE-NPA v5 — telemetry-efficiency results

Status: **passed**.

This was a local, disposable benchmark on the RTX 3060 Ti. No frozen Crucible
artifact, active training run, RunPod job or previous v4 output was modified.

## Integrity

- 10 repeated GPU runs: 5 backends × 2 repetitions.
- 170,233,600-parameter E16, top-2, context 128, Branch B.
- Two warm-up and eight timed optimizer steps per run.
- Exact input-prefix equivalence: passed.
- All losses finite: passed.
- Model-state equivalence against no-telemetry baseline: passed exactly.
- Loss tolerance: passed in every comparison.
- Full telemetry counter equality: passed.
- Primary telemetry counter equality: passed.
- Maximum full-backend floating telemetry delta: `5.96e-8`, below the declared
  `1e-6` reporting tolerance.

The frozen `ResearchMoETelemetry` implementation emits a warning under the
deterministic control because its weighted CUDA `bincount` has no deterministic
PyTorch implementation. This is aggregate-only, outside the gradient path;
the model states and losses were nevertheless identical.

## Mean timing

Values are totals across the eight timed optimizer steps, averaged over two
repetitions.

| Backend | Wall | Forward CUDA | Backward CUDA | Optimizer/clip | Telemetry finalize | Relative to baseline |
|---|---:|---:|---:|---:|---:|---:|
| No telemetry | 15.494 s | 4.713 s | 7.224 s | 495 ms | — | reference |
| Current full | 17.165 s | 6.118 s | 7.230 s | 499 ms | 1.78 ms | +10.8% |
| GPU full batched | 16.279 s | 5.394 s | 7.234 s | 501 ms | 1.59 ms | +5.06% |
| Minimal scientific | 15.539 s | 4.821 s | 7.242 s | 497 ms | 0.82 ms | +0.29% |
| Buffered full | 16.276 s | 5.389 s | 7.231 s | 498 ms | 1.66 ms | +5.05% |

Peak allocated VRAM was approximately 3.19 GiB. CPU batch preparation was only
about 10–11 ms per eight-step window, so it remains irrelevant compared with
neural execution.

## Findings

The full optimized accumulator reduced the current telemetry overhead from
approximately 1.67 s to 0.78 s per eight-step window—about a 53% reduction in
telemetry-induced wall overhead and about a 5.2% end-to-end improvement over
the current full implementation. The minimal primary-science backend was
within about 0.3% of the no-telemetry baseline.

Buffered full reporting had a similar result. Its lower mean than the batched
full run is not sufficient to claim an additional independent speedup with two
repetitions; it demonstrates that deferred report serialization is compatible
with the same full counters. The current runner already performs summary
reporting outside the measured optimizer window, so filesystem buffering is
not the dominant v5 opportunity.

The meaningful optimization is therefore the GPU-side/vectorized accumulation,
especially replacement of repeated expert loops and fragmented operations.
No neural scheduler is justified by this result.

## Profiler evidence

The one-batch profiler probe reported approximately:

- attention module ranges: **10.9 ms CUDA**;
- MoE module ranges: **173.6 ms CUDA**;
- synchronized backward CUDA event: **224.0 ms**;
- synchronized optimizer CUDA event: **49.2 ms**;
- routing/indexing-related CUDA operators: **33.7 ms**.

The routing/indexing total includes dispatch-related `index_put`, `index_add`,
`bincount`, `topk`, gather/scatter and autograd indexing operators; it is not a
pure telemetry-only counter. The evidence indicates that MoE/expert execution
dominates the forward pass, while backward is the largest single measured
training phase. Attention is comparatively small, and optimizer work is
material but secondary. A learned scheduler should not be pursued until
kernel-level MoE dispatch optimization is considered.

## Recommendation

Preserve the frozen telemetry implementation for existing results. For a
future new training package, review the `gpu_full_batched` backend as a
candidate implementation, retaining the full schema and the equivalence
tests. Use `minimal_scientific` when optional diagnostic fields are not needed.
Do not train SSRE-NPA's neural controller yet.

## Accepted output

`full_run_v3/telemetry_efficiency_summary.json`

The complete SHA-256 inventory is in `artifact_hashes.json`.

