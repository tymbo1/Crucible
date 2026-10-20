# GPU Neural SSRE v2 measurement protocol

## Scope

This is a local GPU execution study, not a Crucible scientific training run.
It uses disposable model instances and optimizer updates loaded from immutable
anchors. No checkpoint, manifest, or training result is modified.

## Full-model matrix

The full-model matrix uses:

* frozen E16, top-2 Crucible geometry;
* context packages 128, 256, and 512;
* seeds 42 and 1337;
* the original Python expert/rank loop;
* v1 rank-preserving grouped execution;
* v1 largest-first grouped execution as an ordering control;
* rank-major grouped execution as a distinct dispatch-layout candidate;
* identical optimizer, learning rate, BF16 autocast, deterministic controls,
  schedules, batch assembly and disposable optimizer steps.

Default full timing is two warm-up optimizer steps, eight timed optimizer
steps, and two repetitions per arm and cell. The preflight is deliberately
smaller and is not part of the full matrix.

## Strict equivalence gate

For each candidate, a fresh one-step probe compares:

* per-microbatch logits;
* scalar loss;
* accumulated gradients;
* post-step model state;
* post-step optimizer state, including every tensor in every optimizer state
  slot and optimizer-group metadata.

The accepted v1 tolerances remain in force:

* logits: 5e-3 absolute;
* loss: 1e-5 absolute;
* gradients: 5e-4 absolute;
* post-step model parameters: 5e-4 absolute.

Optimizer-state equality is reported separately and is required for a strict
full-state pass. A configuration with matching model parameters but a
different optimizer-state digest is not called fully equivalent.

## Dispatch fixture

The fixture uses the first N frozen layer-0 expert modules for N in 4, 8,
and 16. It varies token count and three deterministic routing distributions:

* balanced;
* moderate skew;
* high skew.

This isolates execution shape. It does not reproduce a trained E4/E8 model,
and results must not be interpreted as expert-count language-model quality.

Rank-preserving grouping is the strict reference candidate. Combined-rank and
padded-batched paths are diagnostic shape-changing candidates. Their timing
may be useful, but any numerical drift is recorded and excluded from the
strict selector gate.

## Timing

CUDA events measure forward and backward device time. Host wall time includes
Python dispatch and synchronization. Warm-up iterations are excluded from
reported means. No profiler is active during timing.

## Decision rule

The V2 result supports a future neural selector only if at least two
strictly-equivalent execution plans show a repeatable workload-dependent
ordering after repeated paired measurements. If one strict plan dominates,
keep that deterministic plan. A controller is not justified merely because
the workload is irregular.
