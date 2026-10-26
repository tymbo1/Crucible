# Digitwise Addition Competence Ladder

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only

## Objective

Localize the addition failure by separating operand combination from positional composition and carry propagation.

The ladder contains:

1. single-digit addition with no carry;
2. two-digit addition with no carry;
3. later carry stages only if the preceding competence gate passes.

This run is intentionally limited to the first two stages. No carry examples are included.

## Interfaces

- Canonical: `a=3; b=4` or `a=23; b=14`.
- Factorized: separate digit fields, e.g. `a_digit=3; b_digit=4` or `a_tens=2; a_ones=3; b_tens=1; b_ones=4`.

The same operand pairs, targets and seeds are used in both representations.

## Controls and gates

- Seeds `42` and `1337`.
- 500 updates per stage.
- Curriculum arm continues from the previous stage checkpoint.
- Direct arm starts from the same initial model and receives an equal total update budget on the current stage task.
- Training competence gate: at least 95% exact training accuracy.
- Advancement gate: at least 95% training and 90% validation exact accuracy.
- If any Stage 1 arm fails the training competence gate or advancement gate, Stage 2 is not run; the failure is recorded and no carry claim is made.
- Deterministic symbolic oracle must score 100%.
- Training and validation operand pairs are disjoint.
- Every digit and relevant single-digit sum appears in Stage 1 training; every legal digit and positional role appears in Stage 2 training. Under two-digit no-carry constraints, tens digits `9` are impossible because both operands have positive tens digits; the legal tens coverage is `1..8`.
- The original sealed test corpus is not opened.
- No GPU, E4 routing, cloud resources or architecture changes.

## Measurements

Report exact/token accuracy, by-input group and output length, training-fit status, validation transfer, gate status and elapsed CPU time. Single-digit held-out pairs and two-digit held-out pairs remain separate.

## Interpretation

- Stage 1 low train: basic two-operand combination or optimization failure.
- Stage 1 high train/low validation: single-digit transfer failure.
- Stage 1 passes but Stage 2 low train: positional composition/interface difficulty.
- Stage 2 high train/low validation: two-position transfer failure.

No result from this ladder establishes general arithmetic or justifies neuromorphic integration by itself.
