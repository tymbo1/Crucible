# Crucible Two-Candidate Pair-Coverage Experiment v1

Rotated balanced unordered-pair holdouts using the unchanged dense cyclic training path. Validation contexts were represented during training and both layouts used `TRAIN_COMPACT`.

| Partition | Train panel | Held-out panel | Seed | Train | Gate | Validation | Paired validation |
|---|---|---|---:|---:|---|---:|---:|
| P0 | M0 | M1 | 42 | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| P0 | M0 | M1 | 1337 | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| P1 | M1 | M2 | 42 | 144/144 (1.000) | PASS | 65/96 (0.677) | 1/32 |
| P1 | M1 | M2 | 1337 | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| P2 | M2 | M3 | 42 | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| P2 | M2 | M3 | 1337 | 144/144 (1.000) | PASS | 64/96 (0.667) | 0/32 |
| P3 | M3 | M0 | 42 | 144/144 (1.000) | PASS | 87/96 (0.906) | 23/32 |
| P3 | M3 | M0 | 1337 | 144/144 (1.000) | PASS | 88/96 (0.917) | 24/32 |

## Panel-level validation accuracy

| Cell | Held-out panel | Pair accuracy | Ordered-pair rows |
|---|---|---:|---|
| P0_SEED42 | M1 | 80/96 (0.833) | 0,2=24/24, 1,3=24/24, 4,6=16/24, 5,7=16/24 |
| P0_SEED1337 | M1 | 80/96 (0.833) | 0,2=24/24, 1,3=24/24, 4,6=16/24, 5,7=16/24 |
| P1_SEED42 | M2 | 65/96 (0.677) | 0,3=16/24, 1,2=16/24, 4,7=17/24, 5,6=16/24 |
| P1_SEED1337 | M2 | 96/96 (1.000) | 0,3=24/24, 1,2=24/24, 4,7=24/24, 5,6=24/24 |
| P2_SEED42 | M3 | 96/96 (1.000) | 0,4=24/24, 1,5=24/24, 2,6=24/24, 3,7=24/24 |
| P2_SEED1337 | M3 | 64/96 (0.667) | 0,4=16/24, 1,5=16/24, 2,6=16/24, 3,7=16/24 |
| P3_SEED42 | M0 | 87/96 (0.906) | 0,1=24/24, 2,3=16/24, 4,5=24/24, 6,7=23/24 |
| P3_SEED1337 | M0 | 88/96 (0.917) | 0,1=24/24, 2,3=24/24, 4,5=16/24, 6,7=24/24 |

## Decision boundary

A validation result is interpreted only after the 95% training gate. Panel-following errors support exact pair-coverage dependence; repeated identifier-family failures across panels support a more persistent representation effect. These are behavioral interpretations, not neural-mechanism claims.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_PAIR_COVERAGE_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_PAIR_COVERAGE_NOVELTY_AUDIT_20261027.json)
- [Preflight](PREFLIGHT.json)
- [Summary](SUMMARY.json)
- Per-partition generated data, checkpoints, traces and prediction matrices under `data/` and partition cell directories
