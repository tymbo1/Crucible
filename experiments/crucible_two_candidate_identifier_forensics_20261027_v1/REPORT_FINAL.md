# Crucible Two-Candidate Identifier Forensics v1

Post-hoc, read-only analysis of the committed matched-context prediction matrices. No inference, retraining, checkpoint modification or sealed-test access occurred.

## Ordered recombined-pair accuracy

| Seed | Ordered pair | Correct / rows | Accuracy | Error identities (expected → predicted) |
|---:|---|---:|---:|---|
| 42 | `0,3` | 30/30 | 1.000 | none |
| 42 | `1,4` | 26/30 | 0.867 | 4->1: 4 |
| 42 | `2,5` | 20/30 | 0.667 | 2->5: 10 |
| 42 | `3,0` | 30/30 | 1.000 | none |
| 42 | `3,6` | 20/30 | 0.667 | 3->6: 10 |
| 42 | `4,1` | 20/30 | 0.667 | 1->4: 7, 4->1: 3 |
| 42 | `5,2` | 20/30 | 0.667 | 2->5: 10 |
| 42 | `6,3` | 20/30 | 0.667 | 3->6: 10 |
| 1337 | `0,3` | 30/30 | 1.000 | none |
| 1337 | `1,4` | 30/30 | 1.000 | none |
| 1337 | `2,5` | 21/30 | 0.700 | 2->5: 9 |
| 1337 | `3,0` | 30/30 | 1.000 | none |
| 1337 | `3,6` | 20/30 | 0.667 | 6->3: 10 |
| 1337 | `4,1` | 30/30 | 1.000 | none |
| 1337 | `5,2` | 20/30 | 0.667 | 2->5: 10 |
| 1337 | `6,3` | 20/30 | 0.667 | 6->3: 10 |

## Unordered recombined-pair results

| Seed | Unordered pair | Correct / rows | Accuracy | Errors |
|---:|---|---:|---:|---:|
| 42 | `0,3` | 60/60 | 1.000 | 0 |
| 42 | `1,4` | 46/60 | 0.767 | 14 |
| 42 | `2,5` | 40/60 | 0.667 | 20 |
| 42 | `3,6` | 40/60 | 0.667 | 20 |
| 1337 | `0,3` | 60/60 | 1.000 | 0 |
| 1337 | `1,4` | 60/60 | 1.000 | 0 |
| 1337 | `2,5` | 41/60 | 0.683 | 19 |
| 1337 | `3,6` | 40/60 | 0.667 | 20 |

## Target-slot and abstention behavior

| Seed | Regime | Slot 0 | Slot 1 | Abstention | Wrong predicted identities |
|---:|---|---:|---:|---:|---|
| 42 | FAMILIAR | 80/80 | 80/80 | 80/80 | none |
| 42 | RECOMBINED | 57/80 | 49/80 | 80/80 | 1: 7, 4: 7, 5: 20, 6: 20 |
| 1337 | FAMILIAR | 80/80 | 80/80 | 80/80 | none |
| 1337 | RECOMBINED | 61/80 | 60/80 | 80/80 | 3: 20, 5: 19 |

## Reversal consistency

For non-abstention rows, a reversal group holds the source context, unordered pair and authority fixed while swapping candidate order. A correct matching rule should preserve the predicted identifier and flip the selected slot.

| Seed | Regime | Reversal groups | Same predicted identity | Slot flipped | Both correct | Both wrong, same identity |
|---:|---|---:|---:|---:|---:|---:|
| 42 | FAMILIAR | 80 | 80 | 80 | 80 | 0 |
| 42 | RECOMBINED | 80 | 72 | 72 | 49 | 23 |
| 1337 | FAMILIAR | 80 | 80 | 80 | 80 | 0 |
| 1337 | RECOMBINED | 80 | 79 | 79 | 60 | 19 |

## Interpretation boundary

Persistent pair-level errors support a pair-specific behavioral weakness under this distribution. They do not prove that the model literally memorized pairs, identify an embedding mechanism, or establish a general architectural limitation. The matrices contain repeated predictions from two checkpoints and are not independent observations.

## Artifacts

- [Analysis protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_IDENTIFIER_FORENSICS_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_IDENTIFIER_FORENSICS_NOVELTY_AUDIT_20261027.json)
- [Machine-readable pair analysis](PAIR_ANALYSIS.json)
- [Machine-readable error rows](ERROR_ROWS.csv)
