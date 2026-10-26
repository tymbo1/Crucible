# Crucible Primitive Role-Selection Competence — Final Analysis

CPU-only, 12 arms, 500 updates per arm, seeds 42 and 1337, both frozen interfaces. No sealed-test access.

Verification: PASS. All competence gates: FAIL. Conditional transfer: NOT RUN under the stop rule.

## Previous failure audit

The prior operation-conditioned role-binding run had 800 prediction rows and no output-format errors. Its operation-conditioned validation was 50% in every arm; the new audit separates fixed-operation competence from operation selection.

## Competence results

| Condition | Interface | Seed | Train | Validation | Unequal validation | Equal validation | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| S1_SELECT_FIRST | canonical | 42 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S1_SELECT_FIRST | canonical | 1337 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S1_SELECT_FIRST | factorized | 42 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S1_SELECT_FIRST | factorized | 1337 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S2_SELECT_SECOND | canonical | 42 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S2_SELECT_SECOND | canonical | 1337 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S2_SELECT_SECOND | factorized | 42 | 100.0% (40/40) | 100.0% (10/10) | 100.0% (8/8) | 100.0% (2/2) | PASS |
| S2_SELECT_SECOND | factorized | 1337 | 100.0% (40/40) | 90.0% (9/10) | 87.5% (7/8) | 100.0% (2/2) | FAIL |
| S3_JOINT_OPERATION_CONDITIONED | canonical | 42 | 60.0% (48/80) | 60.0% (12/20) | 50.0% (8/16) | 100.0% (4/4) | FAIL |
| S3_JOINT_OPERATION_CONDITIONED | canonical | 1337 | 61.3% (49/80) | 55.0% (11/20) | 50.0% (8/16) | 75.0% (3/4) | FAIL |
| S3_JOINT_OPERATION_CONDITIONED | factorized | 42 | 77.5% (62/80) | 60.0% (12/20) | 50.0% (8/16) | 100.0% (4/4) | FAIL |
| S3_JOINT_OPERATION_CONDITIONED | factorized | 1337 | 60.0% (48/80) | 50.0% (10/20) | 50.0% (8/16) | 50.0% (2/4) | FAIL |

## Interpretation

At least one primitive/condition failed the preregistered competence gate. No conditional transfer stage was run, and no downstream transfer claim is made.

The next diagnostic should inspect the training curves and fixed-operation predictions before increasing compute or changing architecture.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_ROLE_SELECTION_COMPETENCE_PROTOCOL_20251025.md)
- [Previous failure audit](PREVIOUS_FAILURE_AUDIT.md)
- [Analysis](ANALYSIS_FINAL.json)
- [Predictions](PREDICTIONS.csv)
- [Summary](SUMMARY.json)
- [Preflight](PREFLIGHT.json)
- [Training curves](arm_outputs/)

## Post-hoc failure-mode diagnostic

S1 passed all four arms. S2 passed three of four; the only unequal-validation error was one factorized seed-1337 prediction on `(9,0)`.

S3 produced 32/64 correct unequal-validation predictions. All 64/64 predictions were one of the two operands, but 32/32 pair-groups received the same output for `SELECT_FIRST` and `SELECT_SECOND`. There were no output-format or EOS errors. This is behaviorally consistent with operation-identifier neglect or failed operation-conditioned execution, while remaining agnostic about the internal neural mechanism.

The conditional-transfer stage was not run because the preregistered competence gates failed.
