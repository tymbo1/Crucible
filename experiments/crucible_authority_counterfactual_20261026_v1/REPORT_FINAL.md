# Crucible Authority Counterfactual Diagnostic v1 — Exploratory Fixed-Width Run

Read-only evaluation of verified v2 checkpoints. No retraining, architecture change, new split or sealed-test access.

**Interpretation status:** exploratory only. The equal-width construction
padded `authority=LATER` with two spaces, a surface form absent from v2
training. The position-safe raw-label correction in the sibling v2 output is
the accepted counterfactual result.

## Integrity

{
  "architecture_changed": false,
  "invalid_format_rows": 326,
  "pair_rows": 5600,
  "position_invariant_pairs": 5600,
  "prediction_rows": 11200,
  "retraining_performed": false,
  "sealed_test_accessed": false,
  "source_split_audit": {
    "canonical": {
      "authority_sides": {
        "EARLIER": 280,
        "LATER": 280
      },
      "base_id_overlap": 0,
      "base_train_rows": 80,
      "base_validation_rows": 20,
      "counterfactual_pairs": 280,
      "counterfactual_rows": 560,
      "layouts": {
        "AUTHORITY_EARLY": 80,
        "AUTHORITY_MIDDLE": 80,
        "AUTHORITY_NEAR": 80,
        "UNSEEN_CONFLICTING_CUES": 80,
        "UNSEEN_DIGIT_DISTRACTOR": 80,
        "UNSEEN_NEAREST_DECOY": 80,
        "UNSEEN_REVERSED_AUTHORITY": 80
      },
      "operation_orders": {
        "FS": 280,
        "SF": 280
      },
      "position_invariant_pairs": 280
    },
    "factorized": {
      "authority_sides": {
        "EARLIER": 280,
        "LATER": 280
      },
      "base_id_overlap": 0,
      "base_train_rows": 80,
      "base_validation_rows": 20,
      "counterfactual_pairs": 280,
      "counterfactual_rows": 560,
      "layouts": {
        "AUTHORITY_EARLY": 80,
        "AUTHORITY_MIDDLE": 80,
        "AUTHORITY_NEAR": 80,
        "UNSEEN_CONFLICTING_CUES": 80,
        "UNSEEN_DIGIT_DISTRACTOR": 80,
        "UNSEEN_NEAREST_DECOY": 80,
        "UNSEEN_REVERSED_AUTHORITY": 80
      },
      "operation_orders": {
        "FS": 280,
        "SF": 280
      },
      "position_invariant_pairs": 280
    }
  },
  "source_v2_finalization_verified": true,
  "source_v2_sealed_test_accessed": false,
  "target_operation_flips": 5600,
  "unique_pair_keys": 5600,
  "unique_prediction_keys": 11200,
  "valid_executor_mismatches": 0
}

## Counterfactual invariant

Every pair changes only the fixed-width authority designation; operand and operation positions were identical for all exported pairs.

## A3 paired counterfactual results

Counts are counterfactual pairs. `Correct pointer flips` requires both authority variants to select the correct operation and to change selection.

| Layout | Pairs | Unequal | Pointer flips | Correct pointer flips | Output changes | Correct output flips (unequal) |
|---|---:|---:|---:|---:|---:|---:|
| AUTHORITY_EARLY | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| AUTHORITY_MIDDLE | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| AUTHORITY_NEAR | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| UNSEEN_CONFLICTING_CUES | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| UNSEEN_DIGIT_DISTRACTOR | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| UNSEEN_NEAREST_DECOY | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |
| UNSEEN_REVERSED_AUTHORITY | 160 | 128 | 0/160 | 0/160 | 0/160 | 0/128 |

## A3 selection preferences

| Layout | Selected cue 0 | Selected cue 1 | SELECT_FIRST | SELECT_SECOND |
|---|---:|---:|---:|---:|
| AUTHORITY_EARLY | 320 | 0 | 160 | 160 |
| AUTHORITY_MIDDLE | 320 | 0 | 160 | 160 |
| AUTHORITY_NEAR | 320 | 0 | 160 | 160 |
| UNSEEN_CONFLICTING_CUES | 200 | 120 | 280 | 40 |
| UNSEEN_DIGIT_DISTRACTOR | 320 | 0 | 160 | 160 |
| UNSEEN_NEAREST_DECOY | 320 | 0 | 160 | 160 |
| UNSEEN_REVERSED_AUTHORITY | 232 | 88 | 168 | 152 |

## Primary observations

- A pointer change means the selected operation pointer changed when the authority designation changed.
- A correct pointer flip requires both counterfactual variants to select their respective authoritative operation.
- Equal operands are excluded from the discriminating final-output flip denominator because both operations return the same digit.
- These results test synthetic authority cues and do not establish general instruction hierarchy or real-world trustworthiness.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026.md)
- [Analysis](ANALYSIS.json)
- [Predictions](COUNTERFACTUAL_PREDICTIONS.csv)
- [Pair diagnostics](COUNTERFACTUAL_PAIRS.csv)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Finalization](FINALIZATION.json)
