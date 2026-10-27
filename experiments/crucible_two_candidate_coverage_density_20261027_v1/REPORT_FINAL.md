# Crucible Two-Candidate Coverage-Density Experiment v1

Fixed M2 held-out panel; D4 is reused from the immutable crossover and D8/D12 are newly trained with the same 144-row budget.

| Condition | Seed | Source | Train | Gate | Validation | Candidate selection | Paired | Abstention |
|---|---:|---|---:|---|---:|---:|---:|---:|
| D4 | 42 | prior_crossover | 144/144 (1.000) | PASS | 80/96 (0.833) | 48/64 (0.750) | 16/32 (0.500) | 32/32 |
| D4 | 1337 | prior_crossover | 144/144 (1.000) | PASS | 95/96 (0.990) | 63/64 (0.984) | 31/32 (0.969) | 32/32 |
| D8 | 42 | new | 120/144 (0.833) | FAIL | 67/96 (0.698) | 35/64 (0.547) | 3/32 (0.094) | 32/32 |
| D8 | 1337 | new | 138/144 (0.958) | PASS | 77/96 (0.802) | 45/64 (0.703) | 13/32 (0.406) | 32/32 |
| D12 | 42 | new | 144/144 (1.000) | PASS | 64/96 (0.667) | 32/64 (0.500) | 0/32 (0.000) | 32/32 |
| D12 | 1337 | new | 144/144 (1.000) | PASS | 72/96 (0.750) | 40/64 (0.625) | 8/32 (0.250) | 32/32 |

## Pooled condition results

| Condition | Train gates | Validation | Candidate selection | Paired | Abstention |
|---|---:|---:|---:|---:|---:|
| D4 | 2/2 | 175/192 (0.911) | 111/128 (0.867) | 47/64 (0.734) | 64/64 (1.000) |
| D8 | 1/2 | 144/192 (0.750) | 80/128 (0.625) | 16/64 (0.250) | 64/64 (1.000) |
| D12 | 2/2 | 136/192 (0.708) | 72/128 (0.562) | 8/64 (0.125) | 64/64 (1.000) |

## Structural descriptors

| Condition | Edges | Components | Degree sequence | Held-out M2 distances |
|---|---:|---:|---|---|
| D4 | 4 | 4 | `[1, 1, 1, 1, 1, 1, 1, 1]` | `{'0,3': None, '1,2': None, '4,7': None, '5,6': None}` |
| D8 | 8 | 2 | `[2, 2, 2, 2, 2, 2, 2, 2]` | `{'0,3': 2, '1,2': 2, '4,7': 2, '5,6': 2}` |
| D12 | 12 | 1 | `[3, 3, 3, 3, 3, 3, 3, 3]` | `{'0,3': 2, '1,2': 2, '4,7': 2, '5,6': 2}` |

## Interpretation boundary

A result is interpreted as transfer only after the 95% training gate. The fixed M2 endpoint tests pair coverage under this direct identifier-matching task. D4/D8/D12 are nested and change degree and connectivity together with coverage; their graph descriptors are descriptive and cannot establish topology or path-length causality. No variable-cardinality or sealed-test claim is made.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_COVERAGE_DENSITY_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_COVERAGE_DENSITY_NOVELTY_AUDIT_20261027.json)
- [Preflight](PREFLIGHT.json)
- [Summary](SUMMARY.json)
- Fixed validation: `data/M2_validation.jsonl`
- New condition data, checkpoints, traces and prediction matrices are under `data/`, `D8_SEED*` and `D12_SEED*`.
- Immutable D4 source: [`experiments/crucible_two_candidate_pair_crossover_20261027_v1`](../../experiments/crucible_two_candidate_pair_crossover_20261027_v1)
