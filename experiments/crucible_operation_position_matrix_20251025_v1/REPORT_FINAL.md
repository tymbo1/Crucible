# Crucible Operation-Position Matrix — Final Report

CPU-only A0–A3 comparison. A0/A1 are verified existing artifacts; A2/A3 are newly trained. No sealed-test access.

## Results

| Condition | Interface | Seed | Train | Validation | Unequal validation | Train paired | Held-out paired | Correct-and-switched | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0_WORD_ORIGINAL | canonical | 42 | 60.00% | 60.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A0_WORD_ORIGINAL | canonical | 1337 | 61.25% | 55.00% | 50.00% | 1/32 | 0/8 | 0/8 | FAIL |
| A0_WORD_ORIGINAL | factorized | 42 | 77.50% | 60.00% | 50.00% | 14/32 | 0/8 | 0/8 | FAIL |
| A0_WORD_ORIGINAL | factorized | 1337 | 60.00% | 50.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A1_MARKER_ORIGINAL | canonical | 42 | 60.00% | 60.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A1_MARKER_ORIGINAL | canonical | 1337 | 60.00% | 60.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A1_MARKER_ORIGINAL | factorized | 42 | 60.00% | 60.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A1_MARKER_ORIGINAL | factorized | 1337 | 60.00% | 50.00% | 50.00% | 0/32 | 0/8 | 0/8 | FAIL |
| A2_WORD_FIXED | canonical | 42 | 100.00% | 100.00% | 100.00% | 32/32 | 8/8 | 8/8 | PASS |
| A2_WORD_FIXED | canonical | 1337 | 100.00% | 90.00% | 87.50% | 32/32 | 6/8 | 6/8 | FAIL |
| A2_WORD_FIXED | factorized | 42 | 100.00% | 100.00% | 100.00% | 32/32 | 8/8 | 8/8 | PASS |
| A2_WORD_FIXED | factorized | 1337 | 100.00% | 90.00% | 87.50% | 32/32 | 6/8 | 6/8 | FAIL |
| A3_MARKER_FIXED | canonical | 42 | 100.00% | 100.00% | 100.00% | 32/32 | 8/8 | 8/8 | PASS |
| A3_MARKER_FIXED | canonical | 1337 | 100.00% | 95.00% | 93.75% | 32/32 | 7/8 | 7/8 | PASS |
| A3_MARKER_FIXED | factorized | 42 | 100.00% | 90.00% | 87.50% | 32/32 | 6/8 | 6/8 | FAIL |
| A3_MARKER_FIXED | factorized | 1337 | 100.00% | 100.00% | 100.00% | 32/32 | 8/8 | 8/8 | PASS |

Train paired correctness is reported separately from held-out paired correctness. The paired metric uses unequal operand pairs so FIRST and SECOND produce distinguishable outputs.

## Operation-only output sensitivity

These diagnostics hold operands fixed and swap only the operation field. `argmax changes` counts pairs where the predicted output token changes; `both toward correct` counts pairs where each operation's target logit exceeds the other operand's logit.

| Condition | Interface | Seed | Split | Pairs | Argmax changes | First toward | Second toward | Both toward | Mean logit L2 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| A0_WORD_ORIGINAL | canonical | 1337 | train | 32 | 1 | 17 | 16 | 1 | 0.097 |
| A0_WORD_ORIGINAL | canonical | 1337 | validation | 8 | 0 | 4 | 4 | 0 | 0.663 |
| A0_WORD_ORIGINAL | canonical | 42 | train | 32 | 0 | 16 | 16 | 0 | 0.488 |
| A0_WORD_ORIGINAL | canonical | 42 | validation | 8 | 0 | 4 | 4 | 0 | 0.438 |
| A0_WORD_ORIGINAL | factorized | 1337 | train | 32 | 0 | 17 | 15 | 0 | 0.049 |
| A0_WORD_ORIGINAL | factorized | 1337 | validation | 8 | 0 | 4 | 4 | 0 | 0.196 |
| A0_WORD_ORIGINAL | factorized | 42 | train | 32 | 14 | 30 | 16 | 14 | 12.326 |
| A0_WORD_ORIGINAL | factorized | 42 | validation | 8 | 0 | 4 | 4 | 0 | 20.158 |
| A1_MARKER_ORIGINAL | canonical | 1337 | train | 32 | 0 | 16 | 16 | 0 | 0.144 |
| A1_MARKER_ORIGINAL | canonical | 1337 | validation | 8 | 0 | 4 | 4 | 0 | 0.057 |
| A1_MARKER_ORIGINAL | canonical | 42 | train | 32 | 0 | 15 | 17 | 0 | 0.083 |
| A1_MARKER_ORIGINAL | canonical | 42 | validation | 8 | 0 | 4 | 4 | 0 | 0.077 |
| A1_MARKER_ORIGINAL | factorized | 1337 | train | 32 | 0 | 17 | 15 | 0 | 0.084 |
| A1_MARKER_ORIGINAL | factorized | 1337 | validation | 8 | 0 | 4 | 4 | 0 | 0.108 |
| A1_MARKER_ORIGINAL | factorized | 42 | train | 32 | 0 | 16 | 16 | 0 | 0.033 |
| A1_MARKER_ORIGINAL | factorized | 42 | validation | 8 | 0 | 5 | 3 | 0 | 0.085 |
| A2_WORD_FIXED | canonical | 1337 | train | 32 | 32 | 32 | 32 | 32 | 16.065 |
| A2_WORD_FIXED | canonical | 1337 | validation | 8 | 6 | 7 | 7 | 6 | 17.312 |
| A2_WORD_FIXED | canonical | 42 | train | 32 | 32 | 32 | 32 | 32 | 17.732 |
| A2_WORD_FIXED | canonical | 42 | validation | 8 | 8 | 8 | 8 | 8 | 17.142 |
| A2_WORD_FIXED | factorized | 1337 | train | 32 | 32 | 32 | 32 | 32 | 18.570 |
| A2_WORD_FIXED | factorized | 1337 | validation | 8 | 8 | 7 | 7 | 6 | 22.836 |
| A2_WORD_FIXED | factorized | 42 | train | 32 | 32 | 32 | 32 | 32 | 15.038 |
| A2_WORD_FIXED | factorized | 42 | validation | 8 | 8 | 8 | 8 | 8 | 17.326 |
| A3_MARKER_FIXED | canonical | 1337 | train | 32 | 32 | 32 | 32 | 32 | 18.638 |
| A3_MARKER_FIXED | canonical | 1337 | validation | 8 | 7 | 7 | 8 | 7 | 15.742 |
| A3_MARKER_FIXED | canonical | 42 | train | 32 | 32 | 32 | 32 | 32 | 16.284 |
| A3_MARKER_FIXED | canonical | 42 | validation | 8 | 8 | 8 | 8 | 8 | 16.319 |
| A3_MARKER_FIXED | factorized | 1337 | train | 32 | 32 | 32 | 32 | 32 | 19.934 |
| A3_MARKER_FIXED | factorized | 1337 | validation | 8 | 8 | 8 | 8 | 8 | 18.365 |
| A3_MARKER_FIXED | factorized | 42 | train | 32 | 32 | 32 | 32 | 32 | 16.327 |
| A3_MARKER_FIXED | factorized | 42 | validation | 8 | 6 | 7 | 7 | 6 | 22.758 |

## Answer-position probes

The operation-identity probe is fitted on training states and evaluated on disjoint validation states. Probe decodability is not treated as proof of causal use.

| Condition | Interface | Seed | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Final norm |
|---|---|---:|---:|---:|---:|---:|---:|
| A0_WORD_ORIGINAL | canonical | 1337 | 70.00% | 60.00% | 50.00% | 50.00% | 55.00% |
| A0_WORD_ORIGINAL | canonical | 42 | 65.00% | 55.00% | 60.00% | 60.00% | 60.00% |
| A0_WORD_ORIGINAL | factorized | 1337 | 60.00% | 50.00% | 50.00% | 50.00% | 55.00% |
| A0_WORD_ORIGINAL | factorized | 42 | 85.00% | 55.00% | 50.00% | 100.00% | 100.00% |
| A1_MARKER_ORIGINAL | canonical | 1337 | 60.00% | 50.00% | 50.00% | 55.00% | 50.00% |
| A1_MARKER_ORIGINAL | canonical | 42 | 50.00% | 60.00% | 55.00% | 50.00% | 50.00% |
| A1_MARKER_ORIGINAL | factorized | 1337 | 60.00% | 50.00% | 50.00% | 55.00% | 65.00% |
| A1_MARKER_ORIGINAL | factorized | 42 | 70.00% | 55.00% | 50.00% | 65.00% | 50.00% |
| A2_WORD_FIXED | canonical | 1337 | 85.00% | 70.00% | 65.00% | 60.00% | 80.00% |
| A2_WORD_FIXED | canonical | 42 | 100.00% | 95.00% | 95.00% | 85.00% | 85.00% |
| A2_WORD_FIXED | factorized | 1337 | 85.00% | 80.00% | 80.00% | 80.00% | 85.00% |
| A2_WORD_FIXED | factorized | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_MARKER_FIXED | canonical | 1337 | 90.00% | 65.00% | 75.00% | 70.00% | 65.00% |
| A3_MARKER_FIXED | canonical | 42 | 60.00% | 55.00% | 70.00% | 75.00% | 60.00% |
| A3_MARKER_FIXED | factorized | 1337 | 95.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_MARKER_FIXED | factorized | 42 | 65.00% | 60.00% | 55.00% | 65.00% | 65.00% |

## Interpretation boundary

A2/A3 move the operation field to the answer boundary, changing distance, absolute positions and prompt token distributions together. An improvement therefore establishes position/interface sensitivity in this tested setup, not a unique attention-routing mechanism.

## Verification

Source-control prediction reproduction passed for all 8 A0/A1 arms. Verification errors: none.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_OPERATION_POSITION_MATRIX_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Answer-position probes](ANSWER_POSITION_PROBES.json)
- [Operation sensitivity](OUTPUT_SENSITIVITY.json)
- [Finalization](FINALIZATION.json)
