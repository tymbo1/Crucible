# Crucible Equality Foundational-Learning Experiment

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only

## Objective

Test whether acquiring an equality relation on a separate task improves transfer to unseen equal-operand addition, and compare that learned intervention with an explicit equality feature.

## Frozen addition task

The addition task uses the same 22M dense model, byte tokenizer, evaluator, optimizer and 500-update direct-training recipe as the matched diagonal ablation. It uses the same diagonal validation partition:

- 40 training pairs;
- 15 validation pairs;
- special held-out diagonal pairs `(2,2)` and `(3,3)`;
- unchanged addition targets and evaluator;
- fresh initialization seeds `2025` and `2026`.

Both canonical and factorized interfaces are run. No sealed test data or prior trained checkpoint is used.

## Arms

| Arm | Intervention | Purpose |
|---|---|---|
| `A0_original` | Original addition interface | Baseline |
| `A1_equality_pretrained` | 500 equality-task updates, then 500 fresh addition updates | Test transfer of a learned relational primitive |
| `A2_explicit_equality` | Append `equal=0/1` to each addition prompt | Diagnostic upper bound for direct relational access |

### Equality pretraining task

The pretraining task uses the same digit vocabulary but asks only whether two digits are equal. Its targets are binary equality labels, not addition answers. A frozen balanced 80/20 equality split contains both equal and unequal examples in train and validation. Equality pretraining accuracy and compute are reported separately.

For `A1`, the pretrained model weights are retained, but the addition optimizer is reset before the 500 addition updates. This isolates transferred parameters from optimizer-state carryover.

For `A2`, the equality indicator is supplied explicitly. This is a diagnostic upper bound, not evidence that the model learned equality independently.

## Measurements

- equality pretraining train/validation accuracy and steps;
- addition training competence and validation accuracy;
- special diagonal accuracy and unequal validation accuracy;
- per-pair, operand, sum, difference and commutativity results;
- output-token, EOS and full-sequence correctness;
- additional pretraining compute and total compute;
- checkpoint and initial-anchor hashes;
- independent variation across both seeds and interfaces.

## Interpretation

- If `A1` improves held-out diagonal addition over `A0`, a learned equality primitive transferred across tasks.
- If only `A2` improves, explicit relational information helps but equality pretraining did not establish transferable learning.
- If neither improves, investigate variable binding, representation and the arithmetic operation before adding memory or recursion.

All pooled row counts are descriptive; predictions within an arm are correlated. No result establishes a neural mechanism or autonomous self-learning by itself.
