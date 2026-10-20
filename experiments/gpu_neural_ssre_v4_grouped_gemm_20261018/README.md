# GPU Neural SSRE v4 — grouped-GEMM execution-engine preflight

This is an isolated prototype following the V3 metadata result. It tests a
different execution engine rather than another metadata placement:

- route assignments are sorted once on CUDA;
- active expert/rank groups receive GPU-resident prefix-sum offsets;
- the three expert FFN projections use PyTorch grouped BF16 GEMM;
- output contributions are scattered with one device operation;
- there is no Python loop launching one matrix operation per expert.

The accepted V2/V3 rank-preserving implementation remains the reference.
This directory is disposable engineering work only. It does not modify
Crucible checkpoints, manifests, training runs, V2, or V3 artifacts.

The preflight is not a scientific training result. A candidate can proceed to
a full-model benchmark only if its forward outputs, gradients and losses pass
the existing declared tolerances and its timing is measured against the V2
baseline.

