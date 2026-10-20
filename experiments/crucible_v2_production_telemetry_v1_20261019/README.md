# Crucible V2 production-telemetry validation

This is a separate validation package for the V2 rank-preserving grouped MoE
engine with the surgical telemetry correction validated on 2026-10-19.

The package uses a copied manifest and copied code, with the frozen context
data, anchors and schedules exposed through read-only symlinks. The original
context package and all earlier results remain untouched.

The only scientific code change is in `shared/model_core.py`: the two weighted
telemetry `torch.bincount` reductions run on CPU and their float64 results are
copied back to the existing telemetry accumulators. The model, router, expert
weights, loss, optimizer and sample schedule are unchanged.

The 100-step production gate passed exact model, gradient, optimizer-state and
four-layer telemetry comparisons for seeds 42 and 1337. The paired overhead
benchmark passed with finite results and exact final-state hashes.

The sustained 3,125-update run is the next gated artifact. It writes metrics
only in this directory and does not create or modify scientific checkpoints.
