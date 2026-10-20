# GPU Neural SSRE v2 — strategy landscape

This is an isolated, read-only engineering study following the accepted
GPU Neural SSRE v1 deterministic grouped-expert result.

V2 asks whether the best execution plan changes predictably with workload
shape. It does not train a neural controller.

The study has two deliberately separate parts:

1. A full Crucible E16 benchmark across the frozen context-128/256/512
   packages and seeds 42/1337. The strict arms are the immutable Python
   reference, rank-preserving grouped execution, and a rank-major grouped
   execution candidate. The v1 largest-first arm is retained as a control.
2. A dispatch-only fixture using the frozen E16 layer-0 expert weights while
   varying 4/8/16 expert counts, token counts, and balanced/moderately skewed/
   highly skewed assignments. This is an execution-shape study, not a claim
   about a retrained E4 or E8 model.

Every full-model cell performs a disposable optimizer update. The benchmark
hashes model state and optimizer-state tensors separately. Optimizer settings
are recorded, but a matching configuration is not treated as proof that the
optimizer state itself matched.

All output is written below this new directory. Frozen Crucible packages,
anchors, schedules, v1 artifacts, and prior results are read-only inputs.

The neural-controller gate is:

* if exact-equivalent execution strategies exhibit a stable, predictable
  crossover, a small selector may be justified;
* if one exact strategy wins throughout, retain the deterministic engine and
  do not add neural scheduling overhead;
* shape-changing strategies remain diagnostic unless their declared numerical
  contract is explicitly passed.
