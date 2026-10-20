# GPU Neural SSRE v3 — GPU-resident metadata feasibility

This is an isolated follow-up to the sealed GPU Neural SSRE v2 strategy
landscape. V2 remains read-only.

V3 tests one narrow question before any neural controller is considered:

> Can routing metadata be kept on the GPU without changing the grouped MoE
> computation or its numerical contract?

The preflight compares:

1. `cpu_offsets`: the accepted V2 rank-preserving grouped path;
2. `gpu_offsets_scalar`: GPU `bincount`/prefix-sum metadata, with scalar
   segment boundaries read by the Python dispatch loop;
3. `gpu_segment_mask`: GPU-resident pair IDs and masks, avoiding host offset
   materialization but retaining the Python expert loop.

This is a feasibility probe, not a claim of GPU hardware scheduling. It does
not modify Crucible checkpoints, manifests, training runs, or V2 artifacts.

The first run is deliberately a small dispatch-only preflight using frozen
E16 layer-0 expert weights. Full-model execution, CUDA Graphs, custom kernels
and persistent GPU queues remain separate follow-up gates.

