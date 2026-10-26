# Crucible Shared Candidate-Scoring Cardinality Gate v1

Matched fixed-slot versus shared candidate scoring on identifier-based authority selection.

| Variant | Interface | Seed | Train | Validation 2/3 | Transfer 4/5/8 | Transfer abstention | Transfer flips | Transfer paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SLOT_HEAD | canonical | 42 | 384/576 | 202/312 | 73/144 | 48/48 | 23/48 | 0/48 |
| SLOT_HEAD | canonical | 1337 | 388/576 | 199/312 | 86/144 | 48/48 | 10/48 | 0/48 |
| SHARED_SCORER | canonical | 42 | 497/576 | 197/312 | 46/144 | 43/48 | 8/48 | 0/48 |
| SHARED_SCORER | canonical | 1337 | 444/576 | 185/312 | 65/144 | 48/48 | 7/48 | 0/48 |
| SLOT_HEAD | factorized | 42 | 384/576 | 198/312 | 57/144 | 48/48 | 14/48 | 0/48 |
| SLOT_HEAD | factorized | 1337 | 384/576 | 210/312 | 71/144 | 48/48 | 5/48 | 0/48 |
| SHARED_SCORER | factorized | 42 | 481/576 | 213/312 | 43/144 | 32/48 | 7/48 | 1/48 |
| SHARED_SCORER | factorized | 1337 | 449/576 | 149/312 | 50/144 | 45/48 | 26/48 | 0/48 |

## Interpretation

The training competence gate is 95% pointer accuracy. None of the eight
learned arms passed it: fixed-slot arms reached 384--388/576, while
shared-scorer arms reached 444--497/576. Their validation and four/five/eight-
candidate transfer numbers are therefore descriptive only, not clean learned-
transfer evidence.

The deterministic identifier oracle passed 2,064/2,064 rows across training,
validation and transfer, confirming that the task definition, candidate spans
and evaluator are solvable. Because neither learned variant mastered the
training task, this run cannot distinguish candidate extraction from candidate
matching or establish a shared-scorer advantage. Paired counterfactual
correctness remains the primary metric.

The shared scorer and fixed-slot head use the same dense backbone family and training data. This diagnostic changes the readout deliberately; it does not change the dense backbone or tokenizer.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_SHARED_CANDIDATE_SCORING_CARDINALITY_PROTOCOL_20261026.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Data audit](DATA_AUDIT.json)
- [Protocol manifest](PROTOCOL_MANIFEST.json)
- Per-arm checkpoints and traces are under `trained_arms/`
