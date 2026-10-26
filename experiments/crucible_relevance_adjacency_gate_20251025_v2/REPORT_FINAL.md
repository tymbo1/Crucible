# Crucible Relevance and Adjacency Gate — Final Report

Frozen-checkpoint CPU evaluation with native SDPA output and shadow Q/K attention summaries. No retraining, architecture change or sealed-test access.

## Native checkpoint comparison

| Condition | Interface | Seed | Validation | Held-out paired | Instrumentation max abs diff |
|---|---|---:|---:|---:|---:|
| A0_WORD_ORIGINAL | canonical | 42 | 60.00% | 0/8 | 0.00e+00 |
| A0_WORD_ORIGINAL | canonical | 1337 | 55.00% | 0/8 | 0.00e+00 |
| A0_WORD_ORIGINAL | factorized | 42 | 60.00% | 0/8 | 0.00e+00 |
| A0_WORD_ORIGINAL | factorized | 1337 | 50.00% | 0/8 | 0.00e+00 |
| A1_MARKER_ORIGINAL | canonical | 42 | 60.00% | 0/8 | 0.00e+00 |
| A1_MARKER_ORIGINAL | canonical | 1337 | 60.00% | 0/8 | 0.00e+00 |
| A1_MARKER_ORIGINAL | factorized | 42 | 60.00% | 0/8 | 0.00e+00 |
| A1_MARKER_ORIGINAL | factorized | 1337 | 50.00% | 0/8 | 0.00e+00 |
| A2_WORD_FIXED | canonical | 42 | 100.00% | 8/8 | 0.00e+00 |
| A2_WORD_FIXED | canonical | 1337 | 90.00% | 6/8 | 0.00e+00 |
| A2_WORD_FIXED | factorized | 42 | 100.00% | 8/8 | 0.00e+00 |
| A2_WORD_FIXED | factorized | 1337 | 90.00% | 6/8 | 0.00e+00 |
| A3_MARKER_FIXED | canonical | 42 | 100.00% | 8/8 | 0.00e+00 |
| A3_MARKER_FIXED | canonical | 1337 | 95.00% | 7/8 | 0.00e+00 |
| A3_MARKER_FIXED | factorized | 42 | 90.00% | 6/8 | 0.00e+00 |
| A3_MARKER_FIXED | factorized | 1337 | 100.00% | 8/8 | 0.00e+00 |

## A3 frozen prompt variants

| Variant | Mean validation accuracy | Mean held-out paired rate | Arms |
|---|---:|---:|---:|
| A_ADJACENT | 96.25% | 90.62% | 4 |
| B_EARLY_PADDED | 43.75% | 0.00% | 4 |
| C_NEAR_PADDED | 61.25% | 50.00% | 4 |
| D_COMPETING | 96.25% | 90.62% | 4 |
| E_EXPLICIT_AUTHORITY | 41.25% | 0.00% | 4 |

## Interpretation boundary

Native SDPA remains on the output path and the shadow Q/K probabilities are diagnostic only. Attention mass is not proof of a unique causal circuit. The frozen A3 variants measure robustness under prompt distribution shift, not relearning of each variant.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_RELEVANCE_ADJACENCY_GATE_PROTOCOL_20251025_V2.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Attention summaries](ATTENTION_SUMMARIES.json)
- [Attention traces](ATTENTION_TRACES.jsonl)
- [Finalization](FINALIZATION.json)
