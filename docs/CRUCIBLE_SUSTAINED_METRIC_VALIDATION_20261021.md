# Crucible grouped execution: sustained metric validation

Date: 2026-10-21

## Classification

The RTX 4090 3,125-step paired run is classified as **sustained-metric validated**. It is not classified as full-state equivalent because the wrapper failed before persisting terminal model, gradient and optimizer-state hashes.

The failure was in result parsing, not CUDA execution:

```text
ValueError: Duplicate seed 42 ... line 2
```

The metric log contains nine checkpoint rows per seed. The original parser incorrectly required one row per seed. The corrected parser keys rows by `(seed, interval_step_end)`.

## Captured evidence

Both original and grouped arms completed seeds 42 and 1337 through step 3,125. The captured logs contain 18 matched rows each at steps:

`100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125`

All 18 matched checkpoints have:

- identical logged training and validation metrics;
- identical router and layer telemetry;
- maximum numeric delta `0.0`;
- maximum telemetry delta `0.0`;
- zero non-finite interval events;
- no missing checkpoint keys.

At step 3,125, the logged validation BPB values are:

| Seed | BPB |
|---:|---:|
| 42 | 1.0921908517 |
| 1337 | 1.0837670623 |

The reported final interval timing gives a mean reduction of `18.579%` and a throughput multiplier of `1.228x`. This is a logged interval measurement, not an independently reconstructed pod wall clock.

## Relationship to the 100-step exact-state gate

The earlier RTX 4090 100-step gate passed exact model-state, gradient and optimizer-state hash equality for both seeds, with all four layers of telemetry accepted. Therefore the evidence is now:

1. **Short horizon:** exact state equivalence through 100 steps.
2. **Long horizon:** identical recorded metrics and telemetry through 3,125 steps, but no persisted terminal state hashes.

The second result must not be described as bit-for-bit long-horizon state equivalence.

## Engineering decision

Retain rank-preserving grouped execution as an opt-in validated backend for controlled experiments. Keep the original backend as the regression reference until a future sustained run persists terminal state hashes correctly.

No new GPU run or cloud rental is authorized by this report.

## Parser correction

The original source directory is not a Git repository, so no commit hash is claimed. A separate versioned parser package and regression test were created at:

`/home/timbushnell/crucible_sustained_gate_validator_v2_20261021/`

The tests pass `3/3`. The package accepts repeated seeds at different checkpoints and rejects only duplicate `(seed, interval_step_end)` pairs.

## Source and artifact preservation

Frozen source, manifests, checkpoints and captured metric logs were not modified. The captured files and detailed comparison are preserved in:

`/home/timbushnell/runpod_equivalence_3125_results_20261021_v1/`

