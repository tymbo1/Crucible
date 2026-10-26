# Crucible Authority Counterfactual Diagnostic v2

Read-only raw-label counterfactual evaluation of verified v2 checkpoints.

## Result

The primary layout is `UNSEEN_REVERSED_AUTHORITY`. Its authority field appears after both
operation cues and both operands, so the raw `EARLIER`/`LATER` flip changes no
operand or operation positions. The exported invariant passed for all
800 pairs.

## A3 paired counterfactual results

| Group | Pairs | Correct pointer flips | Correct output flips (unequal) |
|---|---:|---:|---:|
| All A3 runs | 160 | 0/160 | 0/128 |

On the 320 A3 rows, selected-operation accuracy was 116/160 for
`EARLIER` authority and 44/160 for `LATER` authority. By operation order,
the model selected the same operation for both members of every pair: for
`FS`, it selected `SELECT_FIRST` in 60/80 pairs and `SELECT_SECOND` in 20/80;
for `SF`, it selected `SELECT_SECOND` in 56/80 and `SELECT_FIRST` in 24/80.
These are descriptive preferences, not evidence of a particular internal
mechanism.

## Notes

- A correct pointer flip requires both authority variants to select the correct operation.
- Equal operands are excluded from the discriminating output-flip denominator.
- The A3 pointer did not change on any of the 160 position-safe pairs; its
  authority-position asymmetry was therefore a static selection preference,
  not a counterfactual response.
- The earlier fixed-width full-layout export is exploratory and is not pooled with this position-safe result.
- This synthetic diagnostic does not establish general instruction hierarchy or real-world trustworthiness.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026_v2.md)
- [Analysis](ANALYSIS.json)
- [Predictions](COUNTERFACTUAL_PREDICTIONS.csv)
- [Pair diagnostics](COUNTERFACTUAL_PAIRS.csv)
- [Finalization](FINALIZATION.json)
