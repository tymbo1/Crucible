# Crucible Two-Candidate Pair-Coverage Crossover v1

Complete ordered train-panel/held-out-panel crossover. The four prior cyclic cells are included by immutable reference; only the eight missing combinations were trained here.

| Train | Held out | Seed | Source | Train | Gate | Validation | Paired validation |
|---|---|---:|---|---:|---|---:|---:|
| M0 | M1 | 42 | prior | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M0 | M1 | 1337 | prior | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M0 | M2 | 42 | new | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M0 | M2 | 1337 | new | 144/144 (1.000) | PASS | 95/96 (0.990) | 31/32 |
| M0 | M3 | 42 | new | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M0 | M3 | 1337 | new | 144/144 (1.000) | PASS | 95/96 (0.990) | 31/32 |
| M1 | M0 | 42 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M1 | M0 | 1337 | new | 144/144 (1.000) | PASS | 79/96 (0.823) | 15/32 |
| M1 | M2 | 42 | prior | 144/144 (1.000) | PASS | 65/96 (0.677) | 1/32 |
| M1 | M2 | 1337 | prior | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M1 | M3 | 42 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M1 | M3 | 1337 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M2 | M0 | 42 | new | 144/144 (1.000) | PASS | 64/96 (0.667) | 0/32 |
| M2 | M0 | 1337 | new | 144/144 (1.000) | PASS | 64/96 (0.667) | 0/32 |
| M2 | M1 | 42 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M2 | M1 | 1337 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M2 | M3 | 42 | prior | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |
| M2 | M3 | 1337 | prior | 144/144 (1.000) | PASS | 64/96 (0.667) | 0/32 |
| M3 | M0 | 42 | prior | 144/144 (1.000) | PASS | 87/96 (0.906) | 23/32 |
| M3 | M0 | 1337 | prior | 144/144 (1.000) | PASS | 88/96 (0.917) | 24/32 |
| M3 | M1 | 42 | new | 144/144 (1.000) | PASS | 82/96 (0.854) | 18/32 |
| M3 | M1 | 1337 | new | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M3 | M2 | 42 | new | 144/144 (1.000) | PASS | 80/96 (0.833) | 16/32 |
| M3 | M2 | 1337 | new | 144/144 (1.000) | PASS | 96/96 (1.000) | 32/32 |

## Validation by held-out panel

| Held-out panel | Rows | Accuracy | Paired groups | Abstention |
|---|---:|---:|---:|---:|
| M0 | 478/576 | 0.830 | 94/192 | 192/192 |
| M1 | 514/576 | 0.892 | 130/192 | 192/192 |
| M2 | 512/576 | 0.889 | 128/192 | 192/192 |
| M3 | 527/576 | 0.915 | 143/192 | 192/192 |

## Interpretation boundary

Interpret validation only after the 95% training gate. Repeated weakness of the same unordered pair across the three training-partner panels would support pair-specific difficulty; changes with the training partner would support coverage-graph dependence; seed reversals under the same panel combination would support optimization or initialization sensitivity. These are behavioral interpretations, not neural-mechanism claims.

## Provenance

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_PAIR_CROSSOVER_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_PAIR_CROSSOVER_NOVELTY_AUDIT_20261027.json)
- Prior immutable run: [`experiments/crucible_two_candidate_pair_coverage_20261027_v1`](../../experiments/crucible_two_candidate_pair_coverage_20261027_v1)
- New arm data, checkpoints, traces and prediction matrices are under the cell directories.
