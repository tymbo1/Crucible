# Crucible Causally Valid Authority Decision Experiment v1

Matched CPU experiment separating late candidate-pointer decisions from causally valid early and post-context decisions.

## Historical frozen reference

The previously accepted frozen A3 late-authority reference remains 0/160 correct two-way pointer flips on the clean unseen layout.

## Paired validation results

Both authority variants must select their respective operations for the same operand pair and operation order.

| Condition | Interface | Seed | Train paired | Validation paired | Validation operation flips |
|---|---|---:|---:|---:|---:|
| D0_LATE_CANDIDATE_POINTER | canonical | 42 | 0/80 | 0/20 | 0/20 |
| D0_LATE_CANDIDATE_POINTER | canonical | 1337 | 0/80 | 0/20 | 0/20 |
| D0_LATE_CANDIDATE_POINTER | factorized | 42 | 0/80 | 0/20 | 0/20 |
| D0_LATE_CANDIDATE_POINTER | factorized | 1337 | 0/80 | 0/20 | 0/20 |
| D1_EARLY_CANDIDATE_POINTER | canonical | 42 | 80/80 | 20/20 | 20/20 |
| D1_EARLY_CANDIDATE_POINTER | canonical | 1337 | 80/80 | 20/20 | 20/20 |
| D1_EARLY_CANDIDATE_POINTER | factorized | 42 | 80/80 | 20/20 | 20/20 |
| D1_EARLY_CANDIDATE_POINTER | factorized | 1337 | 80/80 | 20/20 | 20/20 |
| D2_LATE_FINAL_DECISION | canonical | 42 | 1/80 | 0/20 | 0/20 |
| D2_LATE_FINAL_DECISION | canonical | 1337 | 0/80 | 0/20 | 0/20 |
| D2_LATE_FINAL_DECISION | factorized | 42 | 0/80 | 0/20 | 0/20 |
| D2_LATE_FINAL_DECISION | factorized | 1337 | 0/80 | 0/20 | 0/20 |

## Row-level validation

| Condition | Interface | Seed | A pointer | B pointer | Operation/decision | Output | Format errors |
|---|---|---:|---:|---:|---:|---:|---:|
| D0_LATE_CANDIDATE_POINTER | canonical | 42 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D0_LATE_CANDIDATE_POINTER | canonical | 1337 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D0_LATE_CANDIDATE_POINTER | factorized | 42 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D0_LATE_CANDIDATE_POINTER | factorized | 1337 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D1_EARLY_CANDIDATE_POINTER | canonical | 42 | 40/40 | 40/40 | 40/40 | 40/40 | 0 |
| D1_EARLY_CANDIDATE_POINTER | canonical | 1337 | 40/40 | 40/40 | 40/40 | 40/40 | 0 |
| D1_EARLY_CANDIDATE_POINTER | factorized | 42 | 40/40 | 40/40 | 40/40 | 40/40 | 0 |
| D1_EARLY_CANDIDATE_POINTER | factorized | 1337 | 40/40 | 40/40 | 40/40 | 40/40 | 0 |
| D2_LATE_FINAL_DECISION | canonical | 42 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D2_LATE_FINAL_DECISION | canonical | 1337 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D2_LATE_FINAL_DECISION | factorized | 42 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |
| D2_LATE_FINAL_DECISION | factorized | 1337 | 40/40 | 40/40 | 20/40 | 24/40 | 0 |

## Interpretation

D1 tests whether authority is usable at candidate pointer positions when it precedes them. D2 tests whether a final readout can use authority that appears after the operation candidates. D0 is the matched late-authority candidate-pointer control. A positive result requires counterfactual two-way correctness, not merely high row accuracy.

D1 passed the training and validation competence gates in all four interface/seed arms. D0 failed the training gate in all four arms, as expected for the inaccessible pointer positions. D2 also failed its training competence gate in all four arms (0/80 paired operation competence in three arms and 1/80 in one), so D2 is an unresolved readout-learning failure rather than evidence that a post-context decision state cannot access authority.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_CAUSALLY_VALID_AUTHORITY_DECISION_PROTOCOL_20261026.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Data audit](DATA_AUDIT.json)
- [Protocol manifest](PROTOCOL_MANIFEST.json)
- Per-arm checkpoints and metrics are under `trained_arms/`
