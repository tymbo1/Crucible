# Crucible Matched Primitive-Competence Gate

Date: 2026-10-24  
Status: frozen before training  
Execution: CPU-only dense control; no GPU, cloud or E4 run

## Objective

Determine whether the dense control can learn and transfer a single modulo-ten operation when the supplied intermediate state is explicit, before interpreting two-operation failure as a compositional bottleneck.

## Frozen design

- Architecture: the same dense control and optimizer used by the 2024-10-24 operation-position experiment.
- Model seeds: `42` and `1337`.
- Training updates: exactly `2,000` per arm and seed.
- Metric checkpoints: steps `0`, `500`, `1,000`, `1,500` and `2,000`.
- Checkpoints retained: initial and final model/optimizer states; intermediate metrics retained in `RUNS.jsonl`.
- Prompt prefix and raw-byte tokenizer: matched to the position-gate composition task, including `step2=operation(operand)` syntax and one-digit answer output.
- Original cognitive benchmark and prior experiment directories: unchanged.

## Data partitions

The feature space is the Cartesian product:

`operation ∈ {add, sub, mul}` × `supplied_intermediate ∈ {0,…,9}` × `operand ∈ {1,2,3,4}`.

There are 120 possible contexts. The split is fixed before training:

- Training: 72 contexts.
- Interpolation validation: 28 contexts not present in training, while every operation, intermediate value and operand value is represented in training.
- Generalization holdout: 20 contexts consisting of the withheld operation–operand pairs `mul(4)` and `sub(1)` across all ten supplied intermediate values. Each individual operation and operand value still appears in training, but these cross-factor combinations do not.
- Tiny overfit arm: 8 training contexts, plus an 8-context probe from the interpolation set.

No context is repeated across partitions. The same fixed context partition is used for both model seeds.

## Preregistered gates

1. Tiny-set overfit: final training exact accuracy approximately 100%; this is an optimization/serialization positive control, not evidence of understanding.
2. Interpolation: final exact accuracy must be at least 90% to pass the primitive-transfer gate.
3. Generalization: report exact accuracy, per-operation results and confidence intervals without a preregistered success threshold.

The final step is evaluated without checkpoint selection. The generalization holdout is not used for training or selection.

## Reported measurements

For every checkpoint and split: exact one-digit accuracy, target loss, predicted-digit histogram and per-operation accuracy. Also report record counts, context coverage, oracle verification, split overlap checks, serialization/token alignment, parameter count and wall time.

## Interpretation rules

- Tiny-set success with interpolation failure means memorization or a transfer failure remains.
- Interpolation success with generalization failure localizes the issue to the withheld cross-factor combinations.
- Interpolation success followed by composition failure is evidence for investigating state propagation or chaining.
- Primitive failure does not justify recursive architecture changes or GPU scaling; first audit representation, sampling and supervision.
