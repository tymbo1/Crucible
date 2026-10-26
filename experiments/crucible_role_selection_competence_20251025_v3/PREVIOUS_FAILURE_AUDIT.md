# Previous Variable-Binding Failure Audit

Source: `experiments/crucible_variable_binding_20251025_v1/ROLE_BINDING_PREDICTIONS.csv`  
Rows: 800; arms: 8; audit: PASS.

The prior validation rows contain only unequal operand pairs. The audit therefore cannot use that validation split to test equal-operand role indistinguishability.

| Arm | Train accuracy | Validation accuracy | Val = first | Val = second | Val = either |
|---|---:|---:|---:|---:|---:|
| B1_role_binding_pretrained_canonical_seed2025 | 50/80 | 10/20 | 10 | 10 | 20 |
| B1_role_binding_pretrained_canonical_seed2026 | 50/80 | 10/20 | 8 | 12 | 20 |
| B1_role_binding_pretrained_factorized_seed2025 | 53/80 | 10/20 | 14 | 6 | 20 |
| B1_role_binding_pretrained_factorized_seed2026 | 50/80 | 10/20 | 12 | 8 | 20 |
| B3_binding_operation_conditioned_canonical_seed2025 | 50/80 | 10/20 | 10 | 10 | 20 |
| B3_binding_operation_conditioned_canonical_seed2026 | 50/80 | 10/20 | 8 | 12 | 20 |
| B3_binding_operation_conditioned_factorized_seed2025 | 53/80 | 10/20 | 14 | 6 | 20 |
| B3_binding_operation_conditioned_factorized_seed2026 | 50/80 | 10/20 | 12 | 8 | 20 |

No output-format errors were found in the prior prediction CSV. The operation-conditioned task was therefore undercompetent in the tested regime; it was not a simple EOS/format failure.

The new experiment separates fixed-operation execution from operation-conditioned selection.
