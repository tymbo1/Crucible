# Crucible Variable Binding — Final Analysis

CPU-only, 16 arms, frozen diagonal addition split, 500 addition updates per arm. No sealed-test access.

Addition fit gate: 16/16 arms passed. Role-pretraining train/validation gate: 0/8 pretrained arms passed. Verification: PASS.

## Addition results

| Arm | Interface | Seed | Train | Validation | Diagonal | Unequal |
|---|---|---:|---:|---:|---:|---:|
| B0_original | canonical | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B0_original | canonical | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B0_original | factorized | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B0_original | factorized | 2026 | 97.5% (39/40) | 46.7% (7/15) | 0/2 | 7/13 |
| B1_role_binding_pretrained | canonical | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B1_role_binding_pretrained | canonical | 2026 | 100.0% (40/40) | 53.3% (8/15) | 0/2 | 8/13 |
| B1_role_binding_pretrained | factorized | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B1_role_binding_pretrained | factorized | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B2_role_explicit | canonical | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B2_role_explicit | canonical | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B2_role_explicit | factorized | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B2_role_explicit | factorized | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B3_binding_operation_conditioned | canonical | 2025 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B3_binding_operation_conditioned | canonical | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |
| B3_binding_operation_conditioned | factorized | 2025 | 100.0% (40/40) | 53.3% (8/15) | 0/2 | 8/13 |
| B3_binding_operation_conditioned | factorized | 2026 | 100.0% (40/40) | 60.0% (9/15) | 0/2 | 9/13 |

## Role-binding pretraining

The auxiliary task required selecting the first or second operand on disjoint ordered-pair examples. It contained no addition targets. Pretrained arms reset the addition optimizer before fine-tuning.

| Arm | Interface | Seed | Role train | Role validation | Addition optimizer reset |
|---|---|---:|---:|---:|---|
| B1_role_binding_pretrained | canonical | 2025 | 62.5% (50/80) | 50.0% (10/20) | yes |
| B1_role_binding_pretrained | canonical | 2026 | 62.5% (50/80) | 50.0% (10/20) | yes |
| B1_role_binding_pretrained | factorized | 2025 | 66.2% (53/80) | 50.0% (10/20) | yes |
| B1_role_binding_pretrained | factorized | 2026 | 62.5% (50/80) | 50.0% (10/20) | yes |
| B3_binding_operation_conditioned | canonical | 2025 | 62.5% (50/80) | 50.0% (10/20) | yes |
| B3_binding_operation_conditioned | canonical | 2026 | 62.5% (50/80) | 50.0% (10/20) | yes |
| B3_binding_operation_conditioned | factorized | 2025 | 66.2% (53/80) | 50.0% (10/20) | yes |
| B3_binding_operation_conditioned | factorized | 2026 | 62.5% (50/80) | 50.0% (10/20) | yes |

## Interpretation boundary

Training competence is not held-out transfer. A pretrained arm that fails its role-binding gate is not evidence that a learned binding skill cannot transfer. A role-explicit or operation-conditioned gain would be a representation result, not evidence that the model independently discovered the relation.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_VARIABLE_BINDING_PROTOCOL_20251025.md)
- [Analysis](ANALYSIS_FINAL.json)
- [Addition predictions](PREDICTIONS.csv)
- [Role-binding predictions](ROLE_BINDING_PREDICTIONS.csv)
- [Summary](SUMMARY.json)
- [Preflight](PREFLIGHT.json)
