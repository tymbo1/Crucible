# Crucible Paired-Positional Exposure Gate — Final Report

Compute-matched CPU comparison. No sealed-test access.

## Training arms

| Arm | Interface | Seed | Expanded train exact | Fixed-near paired | Early paired | Middle paired |
|---|---|---:|---:|---:|---:|---:|
| A0_SHUFFLED_EXPOSURE | canonical | 42 | 100.00% | 100.00% | 100.00% | 100.00% |
| A0_SHUFFLED_EXPOSURE | canonical | 1337 | 99.17% | 100.00% | 93.75% | 100.00% |
| A0_SHUFFLED_EXPOSURE | factorized | 42 | 98.33% | 96.88% | 93.75% | 96.88% |
| A0_SHUFFLED_EXPOSURE | factorized | 1337 | 100.00% | 100.00% | 100.00% | 100.00% |
| A1_GROUPED_EXPOSURE | canonical | 42 | 57.92% | 3.12% | 0.00% | 0.00% |
| A1_GROUPED_EXPOSURE | canonical | 1337 | 51.67% | 0.00% | 0.00% | 0.00% |
| A1_GROUPED_EXPOSURE | factorized | 42 | 52.08% | 6.25% | 0.00% | 0.00% |
| A1_GROUPED_EXPOSURE | factorized | 1337 | 48.33% | 0.00% | 0.00% | 0.00% |
| A2_GROUPED_CONSISTENCY | canonical | 42 | 57.92% | 0.00% | 0.00% | 3.12% |
| A2_GROUPED_CONSISTENCY | canonical | 1337 | 47.50% | 0.00% | 0.00% | 0.00% |
| A2_GROUPED_CONSISTENCY | factorized | 42 | 55.00% | 0.00% | 0.00% | 0.00% |
| A2_GROUPED_CONSISTENCY | factorized | 1337 | 57.50% | 0.00% | 0.00% | 0.00% |

## Held-out operand-pair evaluation

| Layout | A0 shuffled | A1 grouped | A2 grouped + consistency | A0 paired | A1 paired | A2 paired |
|---|---:|---:|---:|---:|---:|---:|
| FIXED_NEAR_PADDED | 83.75% | 32.50% | 43.75% | 59.38% | 3.12% | 3.12% |
| EARLY_PADDED | 82.50% | 33.75% | 42.50% | 56.25% | 0.00% | 0.00% |
| MIDDLE_POSITION | 81.25% | 33.75% | 41.25% | 53.12% | 3.12% | 0.00% |
| UNSEEN_BETWEEN_OPERANDS | 46.25% | 30.00% | 40.00% | 9.38% | 0.00% | 0.00% |
| UNSEEN_EARLY_CONTEXT_FIRST | 78.75% | 33.75% | 42.50% | 46.88% | 0.00% | 0.00% |

## Interpretation boundary

A positive A1 result would support aligned exposure as a training intervention. A2 beating A1 under equal exposure would support the consistency objective specifically. Neither result would establish general relational reasoning beyond the tested layouts.

Attention summaries are descriptive shadow Q/K probabilities while native SDPA remains on the output path.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_PAIRED_POSITIONAL_EXPOSURE_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Attention summaries](ATTENTION_SUMMARIES.json)
- [Finalization](FINALIZATION.json)
