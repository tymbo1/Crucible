# Crucible Matched Diagonal Ablation — Final Analysis

This is read-only post-processing. No model was retrained during finalization and no sealed test data was accessed.

Fit gate: 8/8 arms passed. Verification: PASS.

There are four independently initialized arms per condition (two interfaces × two seeds). Individual prediction rows within an arm are correlated, so pooled row counts are descriptive rather than independent-trial estimates.

## Condition results

| Condition | Special holdout accuracy | All validation accuracy | Diagonal validation | Unequal validation |
|---|---:|---:|---:|---:|
| diagonal | 0/8 (0.00%) | 34/60 (56.67%) | 0/8 | 34/52 |
| unequal_role_swapped | 8/8 (100.00%) | 39/60 (65.00%) | 0/0 | 39/60 |

The diagonal and role-swapped unequal special conditions use the same special sum values and the same second-operand frequencies. The first-operand frequency difference is the explicit complementary limitation.

## Conclusions

- The diagonal condition's special held-out pairs are compared directly with the matched unequal special pairs, rather than with the entire validation set alone.
- There is no mixed condition in this role-swapped control.
- Output-token, full-sequence, EOS, operand, sum, difference, pair and commutativity diagnostics are in `ANALYSIS_FINAL.json`.
- No result by itself establishes embedding collision, attention interference or an architectural ceiling.

## Artifacts

- [Final analysis](ANALYSIS_FINAL.json)
- [Final prediction matrix](PREDICTIONS_WITH_FIT.csv)
- [Per-arm JSON/confusion data](forensics/arm_outputs/)
- [Per-arm heatmaps and confusion SVGs](forensics/plots/)
- [Raw report](REPORT.md)
- [Preflight](PREFLIGHT.json)
- [Initial anchors](INITIAL_ANCHORS.json)
