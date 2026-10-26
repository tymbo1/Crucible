# Crucible Digitwise Addition — Matched Diagonal Ablation

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only

## Objective

Test whether the balanced-holdout diagonal failure is specific to equal operands, rather than a consequence of an arbitrary held-out pair set.

This is a behavioral distribution-control experiment. It is not an architecture, optimizer, activation, recursion or neuromorphic experiment.

## Frozen controls

- Existing 22M dense model and initialization procedure.
- Existing byte tokenizer, evaluator, target masking and decoding.
- Existing direct-training recipe and AdamW settings.
- 500 updates per arm.
- Interfaces: `canonical` and `factorized`.
- Fresh initialization seeds: `2025` and `2026`.
- 40 training pairs and 15 validation pairs per condition.
- No curriculum, architecture, tokenizer, optimizer, update-budget or task-definition changes.
- No access to the sealed test corpus or prior model checkpoints during evaluation.

## Frozen conditions

The valid universe is the 55 ordered pairs with `a+b<10`. All three validation sets share the same 13-pair background:

`balanced_a_validation - {(2,2),(3,3)}`.

The two special pairs are then varied:

| Condition | Special validation pairs | Diagonal count | Purpose |
|---|---|---:|---|
| `diagonal` | `(2,2)`, `(3,3)` | 2 | Equal-operand holdout |
| `unequal_matched` | `(2,4)`, `(3,1)` | 0 | Unequal control |
| `mixed` | `(2,2)`, `(2,4)` | 1 | Mixed replication/control |

The `diagonal` and `unequal_matched` conditions have identical special sum values, 4 and 6, and identical special first-operand frequencies `{2,3}`. Their second-operand frequencies cannot also be identical under these sum and non-equality constraints; this unavoidable difference is recorded in the preflight and analysis manifests rather than hidden.

All conditions have identical split sizes and the same background examples. Each condition contains exactly 15 validation and 40 training pairs.

## Preflight requirements

Before training, the runner must verify:

- exactly 55 unique valid pairs per condition;
- 40/15 train/validation counts;
- no train/validation overlap;
- correct labels and oracle sums;
- equal background pair membership across conditions;
- the intended diagonal counts;
- matching special sum multiset for the diagonal and unequal conditions;
- no sealed-test path access.

The runner writes the frozen condition manifest and preflight before any model update.

## Measurements

For every condition, interface and seed, record:

- training and validation exact accuracy;
- the 95% training-competence gate;
- every pair prediction, output token, EOS result and checkpoint hash;
- validation accuracy by condition, individual diagonal pair, first operand, second operand, sum and absolute difference;
- diagonal versus unequal validation accuracy;
- output-token versus full-sequence errors;
- prediction distributions for the previously observed problematic pairs.

Training rows are retained for audit but are not treated as independent held-out evidence.

## Interpretation

- If the diagonal condition remains substantially worse than the unequal condition across fit-competent arms and seeds, an equality-specific transfer weakness becomes more plausible.
- If the conditions are similar, the earlier diagonal contrast was likely driven substantially by coverage or partition structure.
- If the mixed condition lies between them, that supports a graded structural effect; it is not by itself causal evidence.

No outcome establishes embedding collision, attention interference or an architectural ceiling. Those require separate internal-activation controls.

## Stopping and disclosure

If any frozen invariant fails, stop and report it rather than silently changing the task, model, budget or evaluator. The study is exploratory and does not replace the sealed benchmark.
