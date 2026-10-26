# Single-Primitive Transfer Ladder — Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_SINGLE_PRIMITIVE_TRANSFER_LADDER_PROTOCOL_20261024.md`  
Run directory: `experiments/crucible_single_primitive_transfer_ladder_20261024/`

## Outcome

The symbolic positive control passed every tier at 100%. The dense neural control reproduced the simplest training records but failed to transfer the fixed primitive to the two unseen operands. The more demanding tiers were not sufficiently fit within 500 updates, so their validation failures cannot yet be attributed cleanly to operand binding, parameter binding or composition.

## Final exact accuracy

| Tier | Seed | Train | Validation | Interpretation |
|---|---:|---:|---:|---|
| Single fixed | 42 | 100.00% | 0.00% | Clear transfer failure; validation has 2 examples |
| Single fixed | 1337 | 100.00% | 0.00% | Same transfer failure across seed |
| Single operand | 42 | 10.99% | 0.00% | Underfit; no transfer conclusion |
| Single operand | 1337 | 10.99% | 0.00% | Underfit; no transfer conclusion |
| Single parameter | 42 | 65.00% | 0.00% | Incomplete training fit |
| Single parameter | 1337 | 75.00% | 0.00% | Incomplete training fit |
| Depth 2 | 42 | 69.44% | 11.11% | Incomplete training fit |
| Depth 2 | 1337 | 61.11% | 0.00% | Incomplete training fit |
| Depth 3 | 42 | 10.19% | 14.81% | Near-chance/underfit |
| Depth 3 | 1337 | 12.96% | 3.70% | Near-chance/underfit |

## Controls and integrity

- Five tiers × two fixed seeds = 10 CPU arms.
- 500 updates per arm; checkpoints at 0, 100, 250 and 500.
- Symbolic oracle: 100% on every train and validation tier.
- Zero oracle errors.
- Zero train/validation prompt overlaps.
- Multi-operand training covered every digit value.
- No original sealed test set was opened.
- GPU use was disabled.
- `git diff --check` passed.

## Interpretation

The most defensible finding is the simplest one: under the current raw-byte interface and training budget, Crucible can memorize `y=(x+1) mod 10` for `x=0..7` but does not transfer it to `x=8,9`, across both seeds. This is stronger evidence for a neural transfer/generalization problem than for a broken symbolic target generator, but the two-example validation set is too small for a precise accuracy estimate.

The remaining tiers are diagnostic rather than decisive. Their training accuracies were below reliable fit, so longer exposure or a smaller controlled dataset is required before interpreting their validation results as failures of operand diversity, parameter binding or composition.

## Decision

Do not introduce BDH, spiking attention, Hebbian plasticity or GPU scaling as a response to this run. First run a smaller exposure-matched canonical primitive control with more repeated training exposure and a larger held-out operand set. If it still fits the seen operands but fails unseen operands, the next software experiment should compare a factorized/structured representation against the current serialized interface.

Only after single-primitive transfer is established should depth-2/3 results be used to diagnose composition. Neuromorphic and ProjectBrain associative-memory mechanisms remain a later, independent substrate track rather than a presumed fix for this neural execution failure.
