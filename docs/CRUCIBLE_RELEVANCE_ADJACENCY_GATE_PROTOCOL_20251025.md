# Crucible Relevance and Adjacency Gate

Date: 2025-10-25  
Status: preregistered CPU-only frozen-checkpoint gate

## Objective

Test whether the conditional-selection behavior established by A2/A3 remains
effective when the instruction is distant, padded, accompanied by irrelevant
information, or marked as authoritative. Compare the successful A2/A3
checkpoints with the failed A0/A1 controls while keeping all source training
artifacts immutable.

This is a robustness and mechanism experiment. It does not establish explicit
relevance validation, deliberate reasoning, or general instruction following.

## Conditions

Native checkpoint conditions:

| Condition | Encoding | Position |
|---|---|---|
| A0 | Original words | Original position |
| A1 | Length-preserving markers | Original position |
| A2 | Original words | Answer boundary |
| A3 | Length-preserving markers | Answer boundary |

Frozen A3 prompt variants, using the same marker encoding and operand pairs:

| Variant | Manipulation |
|---|---|
| A — adjacent | Original successful A3 prompt |
| B — early padded | Operation moved early with neutral context between it and operands |
| C — near padded | Neutral context added while operation remains adjacent to the answer |
| D — competing | Earlier decoy operation plus the authoritative operation near the answer |
| E — explicit authority | Early operation plus explicit PRIMARY authority and use pointer |

The variants are evaluated only on the frozen A3 checkpoints. They are not
retrained, so their results are distribution-shift robustness results rather
than claims that the model could learn each new format from scratch.

## Measurements

- Training and validation exact output accuracy.
- Unequal-pair paired conditional correctness on training and validation.
- Operation-only output changes and target-logit margins.
- Layer/head attention mass from the answer position to operation, operand,
  decoy and pointer spans.
- Maximum absolute logit difference between native SDPA and the instrumented
  attention implementation.

## Controls and boundaries

- Reload existing A0/A1/A2/A3 checkpoints; do not modify or retrain them.
- Use the existing immutable train/validation records only.
- Do not open the sealed test set.
- Do not alter model weights, architecture, tokenizer, optimizer or training
  artifacts.
- The manual attention implementation must reproduce native logits within the
  preregistered tolerance of `1e-5` maximum absolute error before attention
  summaries are interpreted.
- Attention mass and probe results are mechanistic diagnostics, not proof of a
  unique causal circuit. The operation-swap output test is an input
  intervention holding operands fixed.

## Decision boundary

If A3 remains accurate under padding but fails with competing cues, prioritize
relevance selection/interference. If performance degrades with distance and
neutral padding, prioritize positional propagation. If explicit authority
restores distant performance, test learned source/pointer representations next.
If the instrumented forward pass does not reproduce native logits, stop and
report the instrumentation failure rather than interpreting attention values.
