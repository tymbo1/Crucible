# Crucible Compositional Authority-Transfer Gate v1

Dynamic candidate-pointer gate trained on three candidates and evaluated on held-out three-candidate layouts plus four-candidate transfer.

| Interface | Seed | Train pointer | Validation-3 pointer | Transfer-4 pointer | Transfer-4 abstention | Transfer-4 flips | Transfer-4 paired |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 42 | 736/960 | 414/720 | 18/30 | 6/6 | 0/6 | 0/6 |
| canonical | 1337 | 960/960 | 360/720 | 18/30 | 6/6 | 6/6 | 0/6 |
| factorized | 42 | 960/960 | 522/720 | 18/30 | 6/6 | 6/6 | 0/6 |
| factorized | 1337 | 944/960 | 354/720 | 18/30 | 6/6 | 6/6 | 0/6 |

## Interpretation

The training competence gate is 95% pointer accuracy. Three of four arms
passed it: canonical/1337, factorized/42 and factorized/1337. The
canonical/42 arm remained below the gate at 736/960.

Validation-3 and Transfer-4 are separate: Transfer-4 contains four candidates
although training contains only three. Even the training-competent arms did
not show reliable same-count layout transfer; their Validation-3 pointer
accuracies were 50.0%, 72.5% and 49.2%. Across all four arms, Transfer-4
pointer accuracy was 72/120 (60.0%) and two-way counterfactual correctness was
0/24. Therefore the compositional transfer gate failed.

The failure is structured rather than uniform. Transfer abstention was correct
on 24/24 no-authorized-candidate rows. Relation matching transferred partially
(42/48 across `MATCH_A` and `MATCH_D`), while positional selection transferred
poorly (0/24 `EARLIEST`, 6/24 `LATEST`). These are descriptive diagnostics;
the paired metric remains the primary criterion and did not pass.

This does not establish that the original fixed two-class S4 head scales to
arbitrary candidate counts. The dynamic pointer is a new readout for this
gate; the dense backbone and tokenizer were unchanged, but the readout was
intentionally changed. No sealed test data were accessed.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_COMPOSITIONAL_AUTHORITY_TRANSFER_PROTOCOL_20261026.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Data audit](DATA_AUDIT.json)
- [Protocol manifest](PROTOCOL_MANIFEST.json)
- Per-arm checkpoints and traces are under `trained_arms/`
