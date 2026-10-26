# Crucible Explicit Role-Binding Gate — Final Report

CPU-only parser/executor decomposition. Existing A3 and B0 checkpoints were loaded read-only; no sealed-test access.

## Parser competence

| Interface | Seed | Train tuple accuracy | Validation tuple accuracy |
|---|---:|---:|---:|
| canonical | 42 | 100.00% | 25.00% |
| canonical | 1337 | 99.58% | 10.00% |
| factorized | 42 | 99.58% | 35.00% |
| factorized | 1337 | 100.00% | 15.00% |

## Validation paired correctness

Denominators are pooled over four interface/seed runs per condition; each layout contributes 32 unequal operand-pair groups.

| Condition | Fixed-near | Early | Middle | Unseen-between | Unseen-early-context |
|---|---:|---:|---:|---:|---:|
| C0_ORACLE_PARSER_A3_EXECUTOR | 29/32 | 29/32 | 29/32 | 29/32 | 29/32 |
| C1_LEARNED_PARSER_A3_EXECUTOR | 8/32 | 7/32 | 7/32 | 5/32 | 7/32 |
| C2_LEARNED_PARSER_ORACLE_EXECUTOR | 8/32 | 8/32 | 8/32 | 5/32 | 8/32 |
| C3_RAW_END_TO_END | 7/32 | 1/32 | 6/32 | 0/32 | 1/32 |

## Interpretation

C0 tests sufficiency of oracle canonicalization plus the frozen A3 executor. C1 adds learned tuple extraction while holding execution fixed. C2 tests the same learned tuple against a deterministic executor. C3 is the raw end-to-end B0 control. Oracle parser/executor success is diagnostic scaffolding, not autonomous neural role binding.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_EXPLICIT_ROLE_BINDING_GATE_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Parser predictions](PARSER_PREDICTIONS.csv)
- [Predictions](PREDICTIONS.csv)
- [Parser training traces](PARSER_TRAINING_TRACE.jsonl)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Finalization](FINALIZATION.json)
