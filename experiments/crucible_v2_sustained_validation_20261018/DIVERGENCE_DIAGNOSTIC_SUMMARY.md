# V2 divergence diagnostic summary

Date: 2026-10-19

## Decision

The 3,125-step V2 validation remains blocked. No frozen research run was
modified or restarted.

## Findings

### Strict model-path isolation

With PyTorch deterministic algorithms enabled using `warn_only=False` and the
telemetry class replaced by a no-op diagnostic stub, the unchanged and grouped
V2 paths were each reproducible and terminal-hash identical for both seeds at:

- 10 steps
- 25 steps
- 50 steps
- 100 steps

The checks covered model state, gradients and optimizer state. This rules out
the grouped expert accumulation path as the sole explanation for the earlier
100-step mismatch.

### Telemetry path

Strict deterministic execution with the real telemetry path fails immediately
at `model_core.py` lines 506 and 511 because CUDA `bincount` has no
deterministic implementation. With `warn_only=True`, PyTorch emits the same
warnings. A short ten-step telemetry-enabled diagnostic passed, but the
telemetry-enabled 100-step diagnostic was intentionally stopped after its first
seed because the telemetry synchronization cost was excessive; it is not used
as a result.

The earlier non-strict full telemetry gate failed at step 100 for both seeds.
Therefore the current evidence localizes the problem to the telemetry-enabled
execution contract or its interaction with CUDA scheduling/reduction order,
not to V2 expert accumulation alone. Telemetry is logically read-only; if
enabling it changes training state, that is a contract violation to repair.

## Preserved reports

- `equivalence_100_20261018_retry2/equivalence_report.json` — original full
  telemetry failure.
- `prefix_10_strict_no_telemetry_20261018/prefix_reproducibility_report.json`
- `prefix_25_strict_no_telemetry_20261018/prefix_reproducibility_report.json`
- `prefix_50_strict_no_telemetry_20261018/prefix_reproducibility_report.json`
- `prefix_100_strict_no_telemetry_20261018/prefix_reproducibility_report.json`
- `prefix_10_strict_telemetry_warnonly_20261018/prefix_reproducibility_report.json`

## Required repair before validation

Replace or isolate nondeterministic CUDA telemetry reductions, preferably with
deterministic accumulation that does not participate in the model execution
stream. Re-run the 100-step equivalence gate with the required telemetry
enabled. Do not relax the state or metric criteria and do not launch the
3,125-step validation until that gate passes.
