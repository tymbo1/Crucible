# Crucible Position-Generalization Gate — Final Report

Matched CPU training comparison. No sealed-test access.

## Training arms

| Arm | Interface | Seed | Actual train exact | Actual train paired | Validation field in arm result | Fit gate |
|---|---|---:|---:|---:|---:|---|
| FIXED_POSITION_CONTROL | canonical | 42 | 100.00% | 32/32 | 100.00% | PASS |
| FIXED_POSITION_CONTROL | canonical | 1337 | 100.00% | 32/32 | 100.00% | PASS |
| FIXED_POSITION_CONTROL | factorized | 42 | 100.00% | 32/32 | 100.00% | PASS |
| FIXED_POSITION_CONTROL | factorized | 1337 | 100.00% | 32/32 | 100.00% | PASS |
| POSITION_VARIED_INTERVENTION | canonical | 42 | 95.00% | 28/32 | 65.00% | PASS |
| POSITION_VARIED_INTERVENTION | canonical | 1337 | 97.50% | 30/32 | 55.00% | PASS |
| POSITION_VARIED_INTERVENTION | factorized | 42 | 100.00% | 32/32 | 60.00% | PASS |
| POSITION_VARIED_INTERVENTION | factorized | 1337 | 93.75% | 27/32 | 50.00% | FAIL |

The position-varied training set contains fixed-near, early-padded and middle-position records in a deterministic 27/27/26 split. The actual-training column evaluates those mixed records, rather than substituting a single validation layout.

## Layout evaluation on unseen operand pairs

| Layout | Fixed-control validation | Varied-intervention validation | Fixed paired | Varied paired |
|---|---:|---:|---:|---:|
| A3_EXACT_ADJACENT | 100.00% | 58.75% | 100.00% | 3.12% |
| FIXED_NEAR_PADDED | 100.00% | 57.50% | 100.00% | 3.12% |
| EARLY_PADDED | 40.00% | 53.75% | 0.00% | 0.00% |
| MIDDLE_POSITION | 12.50% | 53.75% | 0.00% | 0.00% |
| UNSEEN_BETWEEN_OPERANDS | 43.75% | 37.50% | 0.00% | 0.00% |
| UNSEEN_EARLY_CONTEXT_FIRST | 60.00% | 53.75% | 0.00% | 0.00% |

## Interpretation boundary

The position-varied intervention did not pass the positional-robustness gate at this training budget: its actual training paired competence was incomplete in aggregate and its held-out paired performance remained near zero. This does not establish that position variation cannot work; it leaves optimization, capacity and curriculum effects unresolved.

Attention summaries are descriptive shadow Q/K probabilities while native SDPA remains on the output path.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_POSITION_GENERALIZATION_GATE_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Actual training metrics](ACTUAL_TRAINING_METRICS.json)
- [Predictions](PREDICTIONS.csv)
- [Attention summaries](ATTENTION_SUMMARIES.json)
- [Finalization](FINALIZATION.json)
