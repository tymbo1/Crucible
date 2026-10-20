# GPU Neural SSRE v4 measurement protocol

## Scope

Measure whether a supported GPU grouped-GEMM primitive can replace the
Python-controlled variable-size expert loop for the frozen E16 layer-0
execution fixture.

## Reference

`cpu_grouped_reference` is V2's accepted rank-preserving grouped path. It
preserves expert/rank segment shapes and the established numerical contract.

## Candidate

`gpu_grouped_gemm` keeps routing and segment order unchanged, but executes:

1. the first expert projection with one grouped GEMM;
2. the second expert projection with one grouped GEMM;
3. the gated activation;
4. the third expert projection with one grouped GEMM;
5. one weighted scatter back to token order.

Empty expert/rank segments are removed on-device. Their active-group offsets
are created with CUDA `bincount` and `cumsum`, and the corresponding expert
weights are selected on CUDA.

## Integrity gate

For balanced and high-skew 512-token fixtures:

- metadata must match the reference assignments;
- forward output max error must be <= `5e-3`;
- expert-gradient max error must be <= `5e-4`;
- all losses must be finite;
- BF16 shape-changing or backend-specific drift must be reported, never
  silently called exact equivalence.

## Timing

Use warmed paired trials with CUDA events for forward/backward device time and
host wall time for dispatch/launch overhead. Do not enable the profiler during
timing. This preflight is intentionally small; no full Crucible training step
is launched here.

## Decision rule

If the grouped-GEMM candidate passes integrity and is faster, prepare a
separate full-model equivalence benchmark. If it fails the declared
equivalence tolerance or is slower, preserve the result as a feasibility
finding and do not introduce a neural selector.

