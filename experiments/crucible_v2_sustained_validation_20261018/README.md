# Crucible V2 sustained validation

This is a separate, read-only validation package for the accepted GPU Neural
SSRE V2 rank-preserving grouped-expert engine.

The wrapper patches freshly constructed models in memory and then delegates to
the frozen context runner. It does not edit the context package, anchors,
manifests, schedules or prior metrics. Outputs belong only in this directory.

## Validation question

Does the V2 grouped MoE execution path remain numerically and scientifically
usable over a sustained Branch-B training run, rather than only over the short
V2 benchmark window?

## Planned matrix

- Frozen context-128 package;
- Branch B, E16, top-2;
- seeds 42 and 1337;
- the immutable manifest learning-rate schedule;
- the immutable 3,125 optimizer-update schedule;
- V2 `grouped_rank_preserving` execution in every MoE layer.

A 100-step mechanical preflight is required before the sustained run. The full
run is launched only after the active RunPod context experiment has completed
and its artifacts have been copied and hashed locally.

No neural scheduler is trained here. No checkpoint is written by the wrapper.
