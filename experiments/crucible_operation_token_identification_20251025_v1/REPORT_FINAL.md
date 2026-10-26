# Crucible Operation-Token Identification — Final Analysis

CPU-only Track A control using the frozen 22M dense architecture, seeds 42/1337, both interfaces, and no sealed-test access.

All gates: PASS. Verification: PASS.

| Interface | Seed | Train | Validation | Gate |
|---|---:|---:|---:|---|
| canonical | 42 | 80/80 | 20/20 | PASS |
| canonical | 1337 | 80/80 | 20/20 | PASS |
| factorized | 42 | 80/80 | 20/20 | PASS |
| factorized | 1337 | 80/80 | 20/20 | PASS |

The target is the operation identifier class, not an operand. Passing this control supports recognition of the explicit identifier but does not establish operation-conditioned execution.

- [Protocol](../../docs/CRUCIBLE_OPERATION_TOKEN_IDENTIFICATION_PROTOCOL_20251025.md)
- [Analysis](ANALYSIS_FINAL.json)
- [Predictions](PREDICTIONS.csv)
- [Training curves](arm_outputs/)
