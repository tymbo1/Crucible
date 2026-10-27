# Track B neural proposer protocol freeze amendment

Date frozen: 2026-10-28  
Status: **FROZEN PENDING SEPARATE EXECUTION APPROVAL**

This amendment resolves the nine execution requirements identified by the
four-arm runner audit. It was created after the original protocol and is not
represented as having been preregistered earlier. It does not add arms, change
the hypotheses, alter the task manifest, weaken the success criteria, or raise
the two-GPU-hour cap.

The machine-readable source is
`CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_FREEZE_AMENDMENT_20261028.json`.

## Frozen decisions

| Requirement | Frozen decision |
|---|---|
| B1 candidate budget | 4 canonical AST candidates per task |
| B1 verifier-call cap | 4 candidate-level calls per task; at most 16 verifier examples and 64 oracle comparisons |
| B2 architecture | 240-input → 64-tanh → 8 fixed-rule logits; top four catalog-ranked candidates |
| B3 architecture | 240-input → 128-tanh → 128-tanh → grammar-masked AST-token logits; depth 3, 7 nodes, 19-token vocabulary, four direct candidates |
| Optimizer | Adam; B2 learning rate 0.01; B3 learning rate 0.003; constant schedule; full training split; 2,000 updates; seeds 42 and 1337 |
| Contradiction fixture | Common externally triggered FIRST→SECOND revision fixture; detection is not claimed |
| Interference fixture | Eight fixed digit-pair probes after every promotion and reset reload |
| Sealed-test source | `EVAL_ORACLE.jsonl`, rows with `split == test`, read only after the final fixed checkpoint |
| Checkpoint rule | Final update 2,000; no early stopping or validation-based checkpoint selection |

## Scientific boundary

The amendment is frozen, but execution is not approved. Both the original
protocol's `run_approved` flag and this amendment's
`scientific_execution_approved` flag must be true, and the independent audit
must pass before any scientific training or test evaluation.

The runner-validation defaults used before this amendment remain validation
fixtures only. They are not retroactively treated as scientific settings.

The amendment also does not authorize the Verified Learning Operator Library
experiment.
