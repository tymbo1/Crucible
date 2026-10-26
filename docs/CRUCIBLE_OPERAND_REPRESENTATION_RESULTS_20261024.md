# Operand Representation and Exposure Control — Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_OPERAND_REPRESENTATION_PROTOCOL_20261024.md`  
Run directory: `experiments/crucible_operand_representation_20261024/`

## Outcome

The canonical numeric and factorized representations both reached 100% training and validation accuracy across both seeds. The opaque-symbol control memorized most training associations but remained near chance on unseen symbols.

## Final exact accuracy

| Arm | Seed | Train | Validation |
|---|---:|---:|---:|
| Canonical numeric | 42 | 100.00% | 100.00% |
| Canonical numeric | 1337 | 100.00% | 100.00% |
| Factorized operands | 42 | 100.00% | 100.00% |
| Factorized operands | 1337 | 100.00% | 100.00% |
| Opaque association | 42 | 96.25% | 5.00% |
| Opaque association | 1337 | 98.75% | 20.00% |

The substantive validation set contains 20 held-out operand values. The opaque arm is a negative control: its symbol-to-value mapping is intentionally unavailable to the model, so its poor transfer is expected and is not an arithmetic failure.

## Controls and integrity

- Three interfaces × two fixed seeds = six CPU arms.
- 1,000 updates per arm; checkpoints at 0, 250, 500, 750 and 1,000.
- Training: 80 operands; validation: 20 disjoint operands from `0..99`.
- Every tens and ones digit occurred in training.
- Symbolic oracle: 100% on every arm and split.
- Zero oracle errors and zero train/validation prompt overlaps.
- No original sealed test set was opened.
- GPU use was disabled.
- `git diff --check` passed.

## Interpretation

This result substantially refines the previous fixed-primitive finding. The earlier experiment trained on `x=0..7` and validated on `x=8,9`; those validation digits were absent from training. In the present control, the held-out operand values are new combinations, but their component digits are represented repeatedly in training. Both canonical and factorized interfaces transfer perfectly under that coverage condition.

Therefore the earlier 0% result cannot be treated as evidence of an unconditional inability to execute `x+1`. It is consistent with a feature-coverage/token-exposure failure, or with the model learning a digit-level rule only when the relevant digit representations have been observed.

The result still does not establish general arithmetic. The task is a small modulo-ten successor with explicit supervision, and the validation split preserves all component digits. A stronger next control should separate component-seen recombination from genuinely unseen digit features and should include carry-sensitive arithmetic.

## Decision

Do not change the neural architecture or introduce BDH/neuromorphic mechanisms based on the earlier failure. The next software gate should be a coverage-controlled arithmetic ladder with:

1. component-seen operand recombinations;
2. held-out operand features absent from training;
3. carry-sensitive multi-digit successor/addition;
4. matched canonical and factorized interfaces.

This run provides positive evidence that representation and coverage materially affect transfer, while the opaque control confirms that arbitrary association memorization remains separable from structured numeric generalization.
