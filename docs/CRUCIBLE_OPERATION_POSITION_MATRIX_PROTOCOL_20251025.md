# Crucible Operation-Position Matrix

Date: 2025-10-25  
Status: preregistered A0–A3 CPU matrix  

## Objective

Test whether placing the operation instruction immediately before the answer
boundary changes conditional operand selection, while separating operation
encoding from operation position.

## Conditions

| Condition | Encoding | Position |
|---|---|---|
| A0 | Original `SELECT_FIRST` / `SELECT_SECOND` words | Original S3 field before operands |
| A1 | Length-preserving one-byte `F` / `S` markers | Original S3 field before operands |
| A2 | Original operation words | Final prompt field immediately before `Answer:` |
| A3 | Length-preserving one-byte markers | Final prompt field immediately before `Answer:` |

A0 is verified from the frozen S3 experiment. A1 is verified from the frozen
single-token marker experiment. Only A2 and A3 are newly trained, using the
same dense 22M model, interfaces, seeds, pair split, optimizer and 500-update
budget.

The A2/A3 fixed prompts retain the original operand text and place the
operation field after the return instruction, immediately before the evaluator
adds the answer boundary. For each operation, A2/A3 prompt byte lengths match
the corresponding A0 prompt, so the output position is held constant by
operation. A3 retains the same neutral padding caveat as A1.

## Primary measurements

- Training and validation exact accuracy.
- Separate operation accuracy and unequal-pair accuracy.
- Paired conditional correctness on the same unequal operand pair.
- Whether predictions switch when only the operation changes.
- Correct-and-switched paired accuracy.
- Layerwise operation-identity probes at the answer position.
- Output-logit sensitivity under operation-only input swaps.

Training paired correctness is reported separately from held-out paired
correctness. No held-out generalization claim is made when the training
competence gate fails.

## Gates and controls

The existing gates remain unchanged: 95% training exact accuracy, 90% overall
validation exact accuracy and 90% unequal validation accuracy. A0/A1 artifact
hashes and metrics must reproduce before A2/A3 results are interpreted.

No sealed test set, architecture modification, additional memory, recursion,
FiLM, extra experts, neural selector or symbolic execution assistance is used.

## Interpretation boundary

An A2/A3 improvement would support position sensitivity of the tested interface,
not prove that attention routing is the mechanism. A null result would motivate
direct attention-path instrumentation or an explicit operation-delivery
intervention, not an immediate architecture expansion.
