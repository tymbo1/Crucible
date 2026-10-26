# Crucible Authority-Aware Role Binding Gate v1 — Corrected Rerun Final Report

CPU-only authority-selection intervention with explicit training distractors. Frozen v3 parser and oracle controls; no sealed-test access.

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
| A1_AUTHORITY_TUPLE_NO_POINTER | canonical | 42 | 97.50% | 0.00% | 0.00% | 8.33% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | canonical | 1337 | 97.50% | 0.00% | 7.50% | 5.00% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | factorized | 42 | 99.17% | 0.00% | 8.75% | 9.58% | 0.00% |
| A1_AUTHORITY_TUPLE_NO_POINTER | factorized | 1337 | 96.25% | 0.00% | 3.75% | 0.42% | 0.00% |
| A2_AUTHORITY_POINTER | canonical | 42 | 95.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| A2_AUTHORITY_POINTER | canonical | 1337 | 84.17% | 100.00% | 100.00% | 100.00% | 100.00% |
| A2_AUTHORITY_POINTER | factorized | 42 | 93.33% | 75.00% | 100.00% | 100.00% | 75.00% |
| A2_AUTHORITY_POINTER | factorized | 1337 | 90.83% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | canonical | 42 | 63.75% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | canonical | 1337 | 84.58% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | factorized | 42 | 25.42% | 100.00% | 100.00% | 100.00% | 100.00% |
| A3_AUTHORITY_POINTER_BALANCED | factorized | 1337 | 37.50% | 100.00% | 100.00% | 100.00% | 100.00% |

## Validation paired correctness

Each cell is correct paired groups / 32 pooled across four interface/seed runs.

| Arm | Authority-near | Authority-early | Authority-middle | Reversed authority | Digit distractor | Nearest decoy | Conflicting cues |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_FROZEN_V3_POINTER | 0/32 | 0/32 | 0/32 | 0/32 | 1/32 | 4/32 | 0/32 |
| A1_AUTHORITY_TUPLE_NO_POINTER | 0/32 | 1/32 | 0/32 | 0/32 | 0/32 | 2/32 | 2/32 |
| A2_AUTHORITY_POINTER | 28/32 | 28/32 | 28/32 | 4/32 | 11/32 | 31/32 | 2/32 |
| A3_AUTHORITY_POINTER_BALANCED | 32/32 | 32/32 | 32/32 | 8/32 | 19/32 | 32/32 | 2/32 |
| E3_ORACLE_BINDING | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

## Interpretation boundary

Authority pointer supervision is evaluated as a learned selection intervention under this synthetic task. It does not establish general authority reasoning or position-independent relevance selection.

## Decision against the preregistered criterion

The primary criterion was at least 90% paired correctness on held-out
conflicting-operation examples, sustained across authority positions and
layouts. The best pointer-supervised arm, A3, achieved 2/32 paired groups
(6.25%) on `UNSEEN_CONFLICTING_CUES`; the criterion therefore failed.

A3 passed the training pointer-competence gate in all four interface/seed
runs (240/240 pointer tuples in each run). On the conflicting-cue validation
rows, its operand pointers remained correct in 80/80 rows, while the
authoritative-operation pointer was correct in 40/80. This localizes the
observed failure in that layout to operation-authority selection within the
tested pipeline, without proving a general mechanism.

The nearest-decoy condition is not balanced evidence of authority reversal:
its protocol fixes the authority side to `EARLIER`. A3 achieved 32/32 there.
The balanced unseen conditions were weaker: 8/32 on reversed authority and
19/32 with digit distractors. The oracle binding control achieved 32/32 on
every validation layout.

The 378 invalid-output rows in the complete prediction export are confined to
non-oracle learned-control outputs; valid rows agree with the deterministic
executor, and EOS is not an independent failure mode in this evaluator.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_AUTHORITY_AWARE_ROLE_BINDING_PROTOCOL_20251025_v2.md)
- [Summary](SUMMARY.json)
- [Analysis](ANALYSIS.json)
- [Predictions](PREDICTIONS.csv)
- [Parser predictions](PARSER_PREDICTIONS.csv)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Finalization](FINALIZATION.json)
