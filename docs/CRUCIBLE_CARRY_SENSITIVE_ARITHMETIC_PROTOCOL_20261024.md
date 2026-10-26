# Carry-Sensitive Arithmetic Gate

Date: 2026-10-24  
Status: frozen before training  
Hardware: CPU only

## Objective

Test whether the structured transfer observed for modulo-ten successor extends to ordinary two-digit addition, and distinguish carry regimes rather than collapsing them into one score.

The task is explicit execution, not rule induction:

`a + b = answer`

with `a,b` in `10..99`.

## Carry regimes

- `no_carry`: no carry from ones to tens and no carry from tens to hundreds; output length is 2.
- `ones_carry`: a carry occurs in the ones place but not in the tens place; output length is 2.
- `cascading_carry`: a ones carry produces a tens carry; output length is 3.

All three regimes are mixed into training. Validation pairs are disjoint from training pairs but use the same regimes.

Pairs with a tens carry but no ones carry are excluded from this first contrast rather than being folded into another category. They should be added as a separate fourth regime in a later arithmetic gate.

## Arms

| Arm | Representation |
|---|---|
| Canonical numeric | `a=27; b=15` |
| Factorized digits | `a_tens=2; a_ones=7; b_tens=1; b_ones=5` |

The same operand pairs, targets, seeds and update counts are used in both arms.

## Neural controls

- Existing dense Stage-A control, unchanged architecture, tokenizer and optimizer.
- Seeds `42` and `1337`.
- 1,000 optimizer updates per arm and seed.
- Checkpoints at steps 0, 250, 500, 750 and 1,000.
- No GPU, E4 routing, original sealed test evaluation or cloud resources.

## Audits and measurements

Before training:

- deterministic symbolic addition oracle must score 100%;
- no train/validation pair overlap;
- all three carry regimes must occur in training and validation;
- every legal digit must occur in each input position in the aggregate training set (tens positions `1..9`, ones positions `0..9`);
- output lengths must be recorded separately.

At each checkpoint and final:

- exact sequence accuracy and token accuracy;
- accuracy by carry regime;
- accuracy by output length;
- training fit versus held-out operand-pair transfer;
- prediction/output-length diagnostics;
- elapsed CPU time and parameter count.

The original sealed test corpus is not opened. Validation is descriptive and is not used for repeated checkpoint selection.

## Interpretation

- If training fit is high and validation is high for all regimes, the model has demonstrated this bounded arithmetic transfer task.
- If no-carry succeeds but carry regimes fail, inspect carry representation and output-length handling.
- If training fit fails, do not attribute validation failure to carry reasoning.
- If canonical and factorized results differ, representation is implicated; if both behave similarly, simple factorization is insufficient to explain the limitation.

This gate does not establish general arithmetic, recursive reasoning, meta-learning or causal competence.
