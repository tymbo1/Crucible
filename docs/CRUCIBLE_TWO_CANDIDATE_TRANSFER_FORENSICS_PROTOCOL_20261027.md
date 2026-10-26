# Crucible Two-Candidate Transfer Forensics v1

Date: 2026-10-27
Status: frozen before execution
Branch: `research/two-candidate-forensics-20261027`

## Objective

Determine whether the validation gap in the accepted two-candidate competence
gate is more consistent with input-layout shift, candidate-pair recombination,
or an interaction between them. This is a checkpoint-only diagnostic. It does
not train, modify or replace any checkpoint.

## Evidence motivating the diagnostic

The accepted gate trains on 144 `TRAIN_COMPACT` rows containing eight ordered
candidate sets:

`(0,1), (1,0), (2,3), (3,2), (4,5), (5,4), (6,7), (7,6)`.

Its 96 validation rows use `VALIDATION_PADDED` inputs containing eight
different ordered candidate sets:

`(0,3), (3,0), (1,4), (4,1), (2,5), (5,2), (3,6), (6,3)`.

All candidate and authority IDs remain within the training vocabulary, but no
validation candidate set is present in training. The two layouts also differ
in byte-level serialization and sequence length.

## Frozen conditions

The diagnostic evaluates the two cyclic checkpoints that passed the original
95% training gate:

- `DENSE_CYCLIC_SEED42`
- `DENSE_CYCLIC_SEED1337`

The four evaluation conditions are:

| Condition | Candidate-pair regime | Input layout | Construction |
|---|---|---|---|
| `T_COMPACT` | Familiar training pairs | Compact | Original 144 training records |
| `T_PADDED` | Familiar training pairs | Padded | Training metadata reserialized with the frozen validation layout |
| `V_COMPACT` | Recombined validation pairs | Compact | Validation metadata reserialized with the frozen training layout |
| `V_PADDED` | Recombined validation pairs | Padded | Original 96 validation records |

Reserialization preserves record ID, candidate IDs, authority ID, target,
counterfactual group and operand metadata. Only the serialized input layout is
changed in the derived conditions. The source records and checkpoints remain
untouched.

## Evaluation and controls

- Reuse the accepted dense `SHARED_SCORER` model and evaluator.
- Load checkpoints exactly as archived by the competence gate.
- Use inference-only evaluation; no optimizer, gradients or parameter writes.
- Verify oracle targets for every condition.
- Reproduce the archived `T_COMPACT` and `V_PADDED` predictions exactly before
  interpreting derived conditions.
- Report row accuracy, abstention, paired counterfactual correctness,
  counterfactual prediction flips, target/prediction confusion, candidate-set
  accuracy and authority-ID accuracy.
- Do not access sealed test data.

## Preregistered interpretation rule

- If `T_PADDED` falls materially below `T_COMPACT` while `V_COMPACT` recovers
  materially above `V_PADDED`, layout shift is implicated.
- If `V_COMPACT` remains close to `V_PADDED` while `T_PADDED` remains close to
  `T_COMPACT`, candidate-pair recombination is implicated.
- If both derived conditions degrade relative to their same-regime controls,
  an interaction is implicated.
- If neither factor changes performance materially, inspect other context or
  representation effects.

These are diagnostic interpretations, not causal proofs. The conditions use
different record populations, so effect sizes are descriptive and must not be
treated as independent validation samples.

## Stop boundary

No new training is permitted in this experiment. If evaluator reproduction,
source invariants or checkpoint loading fails, stop and report the failure
instead of substituting predictions.
