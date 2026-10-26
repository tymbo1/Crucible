# Crucible Paired-Positional Exposure Gate

Date: 2025-10-25  
Status: preregistered CPU-only comparison

## Research question

Does grouping equivalent examples across operation positions, or adding an
explicit cross-position consistency objective, improve transfer to unseen
operand pairs and positional layouts?

The earlier position-varied run remains a historical reference. This protocol
uses a new compute-matched expanded dataset so exposure and optimizer updates
are equal across the three new arms.

## Arms

| Arm | Exposure/order | Objective |
|---|---|---|
| A0 — shuffled exposure | Same three layouts per example, record order shuffled | Task loss only |
| A1 — grouped exposure | Same records, grouped by operand pair and operation across all three layouts | Task loss only |
| A2 — grouped consistency | Same grouped records and compute as A1 | Task loss plus symmetric layout-consistency loss |

The three training layouts are fixed-near, early-padded and middle-position.
Each operand pair contributes both SELECT_FIRST and SELECT_SECOND examples in
all three layouts. Batches contain two complete operand-pair groups (12
records), so A1 and A2 receive aligned positional exposure. A0 receives the
same records and number of token presentations in shuffled order.

The consistency loss is the symmetric KL divergence between the three output
distributions for the same operation and operands, weighted by `0.1`. It does
not receive additional labels or evaluator-only answers.

## Evaluation

Evaluate each checkpoint separately on fixed-near, early-padded and
middle-position layouts, plus unseen-between-operands and unseen
early-context-first layouts. Use the existing disjoint validation operand
pairs and report training and validation paired conditional correctness
separately.

The prior exact A3 checkpoint remains a reference but is not counted as one of
the three new arms.

## Controls and boundaries

- Same dense 22M architecture, tokenizer, initialization seeds, optimizer,
  500 updates, effective batch size and expanded record count.
- No sealed-test access, architecture change, memory, recursion or symbolic
  execution assistance.
- Native SDPA remains on the output path; attention summaries use validated
  shadow Q/K probabilities and are descriptive only.
- A positive result would establish benefit from paired exposure or the
  consistency objective under this synthetic task, not general relational
  reasoning.
