# Crucible Two-Candidate Matched Context Probe v1

Checkpoint-only exploratory probe. Familiar and recombined candidate sets were evaluated on the same source-context strata in fixed `TRAIN_COMPACT` serialization. No training was performed.

## Results by source-context stratum

| Seed | Context source | Familiar row accuracy | Recombined row accuracy | Familiar paired | Recombined paired | Familiar flips | Recombined flips |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | TRAIN_SOURCE | 144/144 (1.000) | 111/144 (0.771) | 48/48 | 15/48 | 48/48 | 15/48 |
| 42 | VALIDATION_SOURCE | 96/96 (1.000) | 75/96 (0.781) | 32/32 | 11/32 | 32/32 | 11/32 |
| 1337 | TRAIN_SOURCE | 144/144 (1.000) | 121/144 (0.840) | 48/48 | 25/48 | 48/48 | 25/48 |
| 1337 | VALIDATION_SOURCE | 96/96 (1.000) | 80/96 (0.833) | 32/32 | 16/32 | 32/32 | 16/32 |

## Overall matched-regime results

| Seed | Familiar | Recombined | Recombined minus familiar |
|---:|---:|---:|---:|
| 42 | 240/240 (1.000) | 186/240 (0.775) | -0.225 |
| 1337 | 240/240 (1.000) | 201/240 (0.838) | -0.162 |

## Interpretation boundary

The same source-context values are used in both candidate regimes within each stratum. This removes the original context-regime confound descriptively, but the probe is generated from source distributions and is not an independent held-out benchmark. It does not establish a general identifier-comparison algorithm or an architectural mechanism.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_MATCHED_CONTEXT_PROBE_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_MATCHED_CONTEXT_PROBE_NOVELTY_AUDIT_20261027.json)
- [Summary](SUMMARY.json)
- [Data audit](DATA_AUDIT.json)
- Per-seed prediction matrices in `PREDICTIONS_SEED42.csv` and `PREDICTIONS_SEED1337.csv`
