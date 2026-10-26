# Crucible Explicit Role-Supervision Gate — Final Report

CPU-only role-supervision intervention. Frozen tuple-only, oracle and raw controls; no sealed-test access.

## Parser training fit

| Interface | Seed | Classifier tuple | Pointer tuple | A position | B position | Operation position |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| canonical | 1337 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| factorized | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| factorized | 1337 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Validation paired output correctness

Denominators are pooled over four interface/seed runs per condition; each layout contributes 32 unequal operand-pair groups.

| Condition | Fixed-near | Early | Middle | Unseen-between | Unseen-early-context | Digit distractor | Competing operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0_TUPLE_ONLY_BASELINE | 8/32 | 8/32 | 8/32 | 5/32 | 8/32 | 3/32 | 2/32 |
| E1_ROLE_SUPERVISED_CLASSIFIER | 15/32 | 16/32 | 17/32 | 12/32 | 15/32 | 5/32 | 9/32 |
| E2_ROLE_SUPERVISED_POINTER | 32/32 | 32/32 | 32/32 | 27/32 | 30/32 | 15/32 | 1/32 |
| E3_ORACLE_BINDING | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| E4_RAW_END_TO_END | 7/32 | 1/32 | 6/32 | 0/32 | 1/32 | 0/32 | 7/32 |

## Interpretation boundary

Role-position supervision is evaluated as a representation/learning intervention. A positive result would support explicit role supervision under this synthetic task; it would not establish general position-independent reasoning.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_EXPLICIT_ROLE_SUPERVISION_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Parser predictions](PARSER_PREDICTIONS.csv)
- [Predictions](PREDICTIONS.csv)
- [Machine-readable analysis](ANALYSIS.json)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Training traces](ROLE_SUPERVISED_TRAINING_TRACE.jsonl)
- [Finalization](FINALIZATION.json)
