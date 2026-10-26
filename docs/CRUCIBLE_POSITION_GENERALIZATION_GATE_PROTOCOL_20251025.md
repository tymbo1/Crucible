# Crucible Position-Generalization Gate

Date: 2025-10-25  
Status: preregistered CPU-only training comparison

## Research question

Does training on varied operation positions improve conditional operand
selection on familiar and unseen positional arrangements, compared with a
matched model trained with the operation fixed immediately before the answer?

## Arms

For each of the canonical and factorized interfaces and seeds 42 and 1337:

| Arm | Training prompt | Purpose |
|---|---|---|
| Fixed-position control | Length-matched neutral context, operation immediately before answer | Isolate a fixed operation position while matching prompt length/components |
| Position-varied intervention | Balanced fixed-near, early-padded and middle-position templates | Train an operation pathway across positions |

Both arms use the same immutable operand-pair split, record count, optimizer,
initialization, 500-update budget, batch size, sample order, operation marker
encoding and interfaces. The fixed control uses neutral context so the two arms
have matched token exposure; the operation remains adjacent to the answer.

## Evaluation layouts

| Layout | Training status for intervention |
|---|---|
| Exact A3 adjacent | Reference format, not used by these new padded arms |
| Fixed near padded | Seen |
| Early padded | Seen |
| Middle position | Seen |
| Between operands | Unseen positional arrangement |
| Early context-first | Unseen positional arrangement |

All evaluations use the existing validation pair set, which is disjoint from
the training pair set. Training-pair evaluations are also reported separately.
No sealed test set is opened.

## Primary measurements

- Output accuracy and unequal-pair paired conditional correctness.
- Correct operation-conditioned output switching.
- Training and validation results separately.
- Performance on unseen positional layouts and unseen operand pairs.
- Shadow Q/K attention summaries using the validated native-SDPA wrapper.

## Controls and interpretation

- No architecture, tokenizer, memory, recursion, FiLM, expert or optimizer
  changes.
- The new control is length-matched rather than byte-identical to the original
  A3 prompt; the frozen A3 checkpoint is retained as the exact-format reference.
- Position-varied success would support learned robustness to the trained
  positional distribution, not position-independent reasoning in general.
- Frozen attention summaries are descriptive and do not establish a unique
  causal circuit.
