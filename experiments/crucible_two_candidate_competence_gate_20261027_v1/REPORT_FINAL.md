# Crucible Two-Candidate Competence Gate v1

The immutable canonical cardinality records were filtered to two candidates only. Validation was evaluated only after an arm passed the 95% training gate.

| Cell | Schedule | Seed | Train pointer | Train paired | Gate | Validation | Validation paired |
|---|---|---:|---:|---:|---|---:|---:|
| DENSE_CYCLIC_SEED42 | cyclic | 42 | 144/144 (1.000) | 48/48 | PASS | 70/96 (0.729) | 9/32 |
| DENSE_SHUFFLED_SEED42 | shuffled | 42 | 132/144 (0.917) | 36/48 | FAIL | not evaluated | not evaluated |
| DENSE_CYCLIC_SEED1337 | cyclic | 1337 | 144/144 (1.000) | 48/48 | PASS | 81/96 (0.844) | 17/32 |
| DENSE_SHUFFLED_SEED1337 | shuffled | 1337 | 120/144 (0.833) | 24/48 | FAIL | not evaluated | not evaluated |

## Class balance

Training has 48 examples for each of target slot 0, target slot 1 and abstention. Validation has 32 examples for each class. No mixed three-candidate rows are included.

## Class-level results

| Cell | Train slot 0 | Train slot 1 | Train abstention | Validation slot 0 | Validation slot 1 | Validation abstention |
|---|---:|---:|---:|---:|---:|---:|
| DENSE_CYCLIC_SEED42 | 48/48 | 48/48 | 48/48 | 16/32 | 22/32 | 32/32 |
| DENSE_SHUFFLED_SEED42 | 42/48 | 42/48 | 48/48 | not evaluated | not evaluated | not evaluated |
| DENSE_CYCLIC_SEED1337 | 48/48 | 48/48 | 48/48 | 25/32 | 24/32 | 32/32 |
| DENSE_SHUFFLED_SEED1337 | 36/48 | 36/48 | 48/48 | not evaluated | not evaluated | not evaluated |

## Interpretation

The isolated two-candidate task is learnable by the dense model under the
cyclic schedule: both seeds pass the 95% training gate and learn both target
slots and abstention perfectly. The shuffled schedule fails the training gate
for both seeds, supporting a schedule/order effect under this fixed budget,
without establishing its optimization mechanism.

The validation results remain a transfer gap. Both passing arms retain perfect
abstention, but paired conditional correctness is only 9/32 and 17/32. Thus
isolating two candidates repairs training competence and removes the global
slot-frequency confound, but it does not yet establish reliable unseen-pair
selection. Variable-cardinality transfer remains closed.

## Interpretation boundary

A training-gate pass establishes competence on this isolated two-candidate distribution. Gated validation is evidence only on the existing two-candidate validation rows; it does not establish transfer to three or more candidates.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_COMPETENCE_GATE_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_COMPETENCE_NOVELTY_AUDIT_20261027.json)
- [Summary](SUMMARY.json)
- [Data audit](DATA_AUDIT.json)
- [Error breakdown](ERROR_BREAKDOWN.json)
- [Reproducibility hashes](HASHES.json)
- [Finalization](FINALIZATION.json)
- Per-arm training and gated validation predictions, traces, results and checkpoints under the cell directories
