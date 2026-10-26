# Crucible Variable-Binding and Role-Conditioned Execution Gate

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only; no sealed-test access

## Objective

Test whether a role-sensitive operand-binding skill transfers to unseen equal-operand addition, and separate learned binding from prompt-level role exposure and operation conditioning.

This is a foundational-learning diagnostic. It is not a memory, recursion, GPU, or autonomous-learning experiment.

## Frozen addition task

The addition task uses the existing 22M dense model, byte tokenizer, optimizer, evaluator, 500-update direct-training budget, and fresh initialization seeds `2025` and `2026` used by the preceding foundational experiments. The diagonal validation split is unchanged:

- 40 training pairs;
- 15 validation pairs;
- special held-out pairs `(2,2)` and `(3,3)`;
- targets are unchanged decimal sums with `a+b<10`;
- no sealed test data or prior trained checkpoint is opened.

Both `canonical` and `factorized` interfaces are evaluated.

## Arms

| Arm | Intervention | Purpose |
|---|---|---|
| `B0_original` | Original addition prompt | Frozen baseline |
| `B1_role_binding_pretrained` | 500 role-binding updates, reset addition optimizer, then original addition prompt for 500 updates | Does a learned role-selection skill transfer? |
| `B2_role_explicit` | Role-explicit addition prompt, no pretraining | Does clearer role serialization help directly? |
| `B3_binding_operation_conditioned` | 500 role-binding updates, reset addition optimizer, then explicit operation-plus-role prompt for 500 updates | Do binding and operation-conditioned execution cooperate? |

The architecture, parameter count, tokenizer, optimizer, learning rate, addition step budget, evaluator, and initialization-anchor procedure are otherwise unchanged. The added pretraining compute for `B1` and `B3` is reported separately.

## Role-binding pretraining task

The auxiliary task contains two operation-conditioned queries:

- `SELECT_FIRST`: return the value bound to the first operand role;
- `SELECT_SECOND`: return the value bound to the second operand role.

The task uses 40 frozen training operand pairs and 10 disjoint validation operand pairs, with both selection operations for every pair (80/20 records). Every digit appears in both operand roles in the training and validation pair sets. The validation pairs are:

`(0,9), (1,8), (2,7), (3,6), (4,0), (5,1), (6,2), (7,3), (8,4), (9,5)`.

The training pairs are the ten equal pairs, both directions of adjacent pairs, and both directions of selected distance-two pairs, excluding the validation pairs. The auxiliary targets are selected operand digits, never addition answers.

The role-binding competence gate is 95% exact accuracy on both training and validation. If it fails, the corresponding transfer result is reported but is not interpreted as evidence against transfer from a learned binding skill.

## Addition prompt interventions

`B0` and `B1` use the frozen original canonical/factorized prompts.

`B2` replaces the operand labels with explicit role slots while retaining the natural-language addition instruction:

`Add integers with no carry. first_operand=...; second_operand=.... Return digits only.`

`B3` uses an explicit operation token and role slots:

`operation=ADD_NO_CARRY; first_operand=...; second_operand=.... Return digits only.`

The `B2` and `B3` prompt changes are diagnostic representation interventions, not evidence of autonomous role discovery.

## Measurements and safeguards

- role-binding pretraining train/validation exact accuracy and fit-gate status;
- addition training competence and validation exact accuracy;
- special diagonal, unequal, and overall validation accuracy;
- per-pair predictions, operand-role and sum breakdowns;
- output-token, EOS, full-sequence and format correctness;
- checkpoint, initial-anchor and immutable data hashes;
- addition optimizer reset verification for pretrained arms;
- separate seed/interface variation;
- no test-set access and no modification of earlier artifacts.

Training rows are competence evidence, not independent held-out observations. Pooled predictions are descriptive because pairs, seeds, and interfaces are correlated.

## Decision criteria

- If role-binding pretraining passes its own gate and improves held-out diagonal addition over `B0`, that supports transfer of a learned role-sensitive skill.
- If `B2` improves but `B1` does not, prompt-level role exposure helps in this setting but learned pretraining did not transfer.
- If `B3` improves beyond both `B1` and `B2`, the combination is a candidate for a follow-up controlled ablation.
- If none improves, the experiment supports neither equality-only nor short role-binding transfer as a sufficient intervention. Investigate variable binding and relation-conditioned execution more directly before adding memory or recursion.

No result from this gate establishes a neural mechanism, general arithmetic competence, or autonomous self-learning.
