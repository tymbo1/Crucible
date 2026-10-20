# GPU Neural SSRE v0 — E16 execution profile

This is a separate, read-only engineering track for GPU-side sparse-expert
execution. It consumes the frozen Crucible E16 Branch-B manifest, seed anchor,
schedule and input records without modifying any of them.

The first gate is attribution, not optimization:

- record the actual token count sent to every expert;
- measure nested MoE, expert and attention ranges;
- inspect CUDA operator time for matrix, routing/indexing, layout/copy and
  launch-related operations;
- measure one warmed forward/backward workload using the frozen input order;
- verify the model state is unchanged.

The probe performs no optimizer step and writes only to its new output
directory. Its timings are attribution measurements, not a training result.

The next gate, which is deliberately not included here, is a deterministic
grouped-expert implementation benchmark. A neural execution-plan controller
must beat that baseline after its own overhead is included.

