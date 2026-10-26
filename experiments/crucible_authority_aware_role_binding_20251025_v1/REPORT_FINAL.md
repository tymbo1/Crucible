# Crucible Authority-Aware Role Binding Gate v1 — Final Report

CPU-only authority-selection intervention. Frozen v3 parser and oracle controls; no sealed-test access.

## v3 baseline reproduction

{
  "checkpoint_hashes": true,
  "paired_validation": {
    "EARLY_PADDED": [
      32,
      32
    ],
    "FIXED_NEAR_PADDED": [
      32,
      32
    ],
    "MIDDLE_POSITION": [
      32,
      32
    ],
    "UNSEEN_BETWEEN_OPERANDS": [
      27,
      32
    ],
    "UNSEEN_COMPETING_OPERATION": [
      1,
      32
    ],
    "UNSEEN_DIGIT_DISTRACTOR": [
      15,
      32
    ],
    "UNSEEN_EARLY_CONTEXT_FIRST": [
      30,
      32
    ]
  },
  "passed": true,
  "sealed_test_accessed": false,
  "source_finalization": "experiments/crucible_explicit_role_supervision_20251025_v3/FINALIZATION.json"
}

## Training fit

| Arm | Interface | Seed | Classifier tuple | Pointer tuple | A position | B position | Operation position |
|---|---|---:|---:|---:|---:|---:|---:|
| A1_AUTHORITY_TUPLE_NO_POINTER | canonical | 42 | 91.67% | 0.00% | 0.00% | 18.75% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | canonical | 1337 | 90.42% | 0.00% | 0.00% | 0.00% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | factorized | 42 | 97.08% | 0.00% | 0.00% | 23.33% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | factorized | 1337 | 91.25% | 0.00% | 0.00% | 10.83% | 0.00% |
| A2_AUTHORITY_POINTER | canonical | 42 | 98.33% | 100.00% | 100.00% | 100.00% | 100.00% |
| A2_AUTHORITY_POINTER | canonical | 1337 | 95.42% | 58.33% | 100.00% | 100.00% | 58.33% |
| A2_AUTHORITY_POINTER | factorized | 42 | 85.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| A2_AUTHORITY_POINTER | factorized | 1337 | 96.25% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | canonical | 42 | 25.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | canonical | 1337 | 80.42% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | factorized | 42 | 87.08% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | factorized | 1337 | 98.33% | 100.00% | 100.00% | 100.00% | 100.00% |

## Validation paired correctness

Each cell is correct paired groups / 32 pooled across four interface/seed runs.

| Arm | Authority-near | Authority-early | Authority-middle | Reversed authority | Digit distractor | Nearest decoy | Conflicting cues |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_FROZEN_V3_POINTER | 2/32 | 0/32 | 0/32 | 0/32 | 1/32 | 4/32 | 0/32 |
| A1_AUTHORITY_TUPLE_NO_POINTER | 2/32 | 3/32 | 2/32 | 1/32 | 0/32 | 1/32 | 2/32 |
| A2_AUTHORITY_POINTER | 24/32 | 24/32 | 28/32 | 1/32 | 2/32 | 24/32 | 2/32 |
| A3_AUTHORITY_POINTER_BALANCED | 32/32 | 32/32 | 32/32 | 4/32 | 3/32 | 32/32 | 4/32 |
| E3_ORACLE_BINDING | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

## Interpretation boundary

Authority pointer supervision is evaluated as a learned selection intervention under this synthetic task. It does not establish general authority reasoning or position-independent relevance selection.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_AUTHORITY_AWARE_ROLE_BINDING_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Analysis](ANALYSIS.json)
- [Predictions](PREDICTIONS.csv)
- [Parser predictions](PARSER_PREDICTIONS.csv)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Finalization](FINALIZATION.json)
