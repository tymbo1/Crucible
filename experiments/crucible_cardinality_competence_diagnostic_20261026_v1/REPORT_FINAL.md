# Crucible Cardinality Competence Diagnostic v1

Training-competence-only representation-by-schedule diagnostic on the frozen canonical cardinality training split.

| Cell | Representation | Schedule | Train pointer | Paired groups | Gate |
|---|---|---|---:|---:|---|
| D0_DENSE_CYCLIC | dense | cyclic | 497/576 (0.863) | 113/192 | FAIL |
| D1_DENSE_SHUFFLED | dense | shuffled | 384/576 (0.667) | 0/192 | FAIL |
| D2_STRUCTURED_CYCLIC | structured | cyclic | 390/576 (0.677) | 6/192 | FAIL |
| D3_STRUCTURED_SHUFFLED | structured | shuffled | 432/576 (0.750) | 48/192 | FAIL |

## Error structure

The abstention subtask was solved in all four cells: 192/192 absent-query
rows were correct in every cell. The present-query rows were the bottleneck:

| Cell | Present-query accuracy | Slot 0 | Slot 1 | Slot 2 |
|---|---:|---:|---:|---:|
| D0 dense/cyclic | 305/384 (79.43%) | 143/192 | 18/48 | 144/144 |
| D1 dense/shuffled | 192/384 (50.00%) | 78/192 | 48/48 | 66/144 |
| D2 structured/cyclic | 198/384 (51.56%) | 54/192 | 0/48 | 144/144 |
| D3 structured/shuffled | 240/384 (62.50%) | 108/192 | 0/48 | 132/144 |

The dense cyclic cell exactly reproduces the prior canonical shared-scorer
training result, 497/576. Shuffling reduced dense training fit to 384/576 and
eliminated paired correctness (0/192). Within the smaller structured model,
shuffling improved fit from 390/576 to 432/576, but neither structured cell
passed and neither learned any of the 48 slot-1 targets.

These are training-distribution observations, not transfer evidence. The
structured model has 3,586 parameters versus 22,106,906 in the dense model, so
the representation comparison is not parameter-matched. The within-
representation schedule contrasts are more interpretable than the
cross-representation contrasts, but they use one seed only.

## Decision

The diagnostic provides evidence that schedule/order affects competence under
this task and budget, but it does not identify schedule as the sole cause and
does not show that explicit features solve the task. All four cells failed the
95% gate. Cardinality transfer remains closed.

The next competence repair should simplify the training target before adding
architectural complexity: use a preregistered two-candidate matching task
with balanced slot-0/slot-1 positives and a separately reported abstention
control, then require full training competence before reintroducing mixed
2/3-cardinality batches. This diagnostic does not run that follow-up.

## Interpretation boundary

This run evaluates only training competence. It does not test validation, cardinality transfer, candidate-count generalization or any sealed test set. A passed cell establishes that its representation and schedule can fit this training distribution; it does not establish transfer.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_CARDINALITY_COMPETENCE_DIAGNOSTIC_PROTOCOL_20261026.md)
- [Novelty audit](../../docs/CRUCIBLE_CARDINALITY_COMPETENCE_NOVELTY_AUDIT_20261026.json)
- [Summary](SUMMARY.json)
- [Data audit](DATA_AUDIT.json)
- [Error breakdown](ERROR_BREAKDOWN.json)
- [Analysis tool](../../tools/analyze_crucible_cardinality_competence_diagnostic_20261026.py)
- Per-cell predictions, traces, results and checkpoints under the cell directories
