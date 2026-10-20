# SSRE-NPA v4 measurement protocol

## Fixed workload

- Frozen context-128 E16 Branch-B manifest.
- Frozen seed-42 anchor and schedule.
- First `warmup_steps + timed_steps` optimizer groups from the schedule.
- Same source indices, batch boundaries, target masks, LR warmup values,
  optimizer geometry and model initialization for every arm.
- Fresh disposable model/optimizer per arm, telemetry mode and repetition.

## Timing phases

1. CPU batch preparation.
2. Synchronous host-to-device `.to(device)` transfer.
3. Forward host wall time and CUDA event time.
4. Backward host wall time and CUDA event time.
5. Gradient clipping plus optimizer step.
6. Telemetry finalization separately, with telemetry-on versus telemetry-off
   forward timing as the primary telemetry-cost contrast.

CUDA is synchronized before/after each measured phase. Initialization,
manifest/source loading, anchor loading and state hashing are outside the
steady-state phase totals and reported separately or as integrity metadata.

## Equivalence gate

The benchmark verifies CPU batch fingerprints for a fixed prefix before GPU
work. For each repetition and telemetry mode it compares baseline/indexed loss
trajectories and final model-state hashes. A change is not accepted as a pure
delivery optimization if input digests, masks, losses or final state diverge
beyond the declared tolerance.

