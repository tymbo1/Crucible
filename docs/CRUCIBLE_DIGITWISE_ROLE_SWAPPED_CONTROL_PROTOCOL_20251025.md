# Crucible Digitwise Addition — Role-Swapped Unequal Control

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only

## Objective

Provide a complementary control for the matched diagonal-ablation result by placing the diagonal operand identities in the second position rather than the first.

## Frozen controls

- Same 22M dense architecture, tokenizer, evaluator, optimizer and 500-update recipe as the matched diagonal ablation.
- Same interfaces: `canonical` and `factorized`.
- Same initialization seeds: `2025` and `2026`, with deterministic anchors regenerated from the frozen model definition and hash-checked.
- Same 40/15 split sizes and same 13-pair background as the matched diagonal study.
- No sealed-test access, architecture changes or prior checkpoint reuse.

## Conditions

| Condition | Special validation pairs | Purpose |
|---|---|---|
| `diagonal` | `(2,2)`, `(3,3)` | Equal-operand reference |
| `unequal_role_swapped` | `(4,2)`, `(1,3)` | Unequal control with the diagonal second-operand identities |

Both special sets have sums `{4,6}`. The diagonal set has second-operand frequencies `{2,3}`, and the role-swapped unequal set has the same second-operand frequencies. First-operand frequencies necessarily differ; this is the complementary limitation to the preceding control.

The common background is the prior 13-pair balanced background, so each condition has 40 training and 15 validation pairs.

## Interpretation

If the diagonal special pairs remain poor while the role-swapped unequal pairs generalize substantially better across both interfaces and seeds, the behavioral equality-transfer baseline is stronger. If both controls degrade, operand-role distribution remains a viable explanation.

The result does not establish a neural mechanism. Training rows are not independent held-out trials, and the two operand marginals cannot be matched simultaneously under the chosen sum constraints.
