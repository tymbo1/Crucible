# GPU Neural SSRE v1 — deterministic grouped-expert baseline

This is an isolated execution experiment built on the frozen Crucible E16
Branch-B model. It preserves v0 and all frozen Crucible artifacts.

Variants:

- `original`: the immutable Python expert/rank loop;
- `grouped_rank_preserving`: GPU token grouping by expert and routing rank,
  preserving the frozen expert input shapes and expert order;
- `grouped_rank_largest_first`: the same rank-preserving representation, but
  executes the largest expert groups first;
- `grouped_sorted`: one expert call per expert, combining both routing ranks;
- `batched_padded`: one padded batched SwiGLU computation over all experts.

The benchmark first runs an equivalence probe for every grouped variant, then
performs warmed repeated forward/backward/optimizer trials with profiling
disabled. It does not train a scientific model or alter any checkpoint. The
optimizer steps are disposable benchmark updates used only to expose backward
and optimizer costs and to test state equivalence.

The neural scheduler is deliberately absent. A later controller must beat the
best deterministic variant including its own inference overhead.

