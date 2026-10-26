# Crucible Causally Valid Authority Decision Experiment v1

Date: 2026-10-26  
Status: preregistered CPU experiment

## Objective

Separate information availability from authority-to-operation binding using
matched counterfactual examples:

- **D0 — late authority / candidate pointer:** authority is after both
  operation candidates; the existing operation-pointer head scores candidate
  positions.
- **D1 — early authority / candidate pointer:** authority is before both
  operation candidates; the same pointer readout must select the operation.
- **D2 — late authority / final decision:** authority remains late, but a new
  two-class decision head reads a final decision token after the complete
  context.

## Frozen task and matched data

Use the existing canonical and factorized base operand splits, with the same
unique training and validation operand pairs for all three conditions. Each
pair is expanded into both operation orders (`FS`, `SF`) and both authority
codes (`E`, `L`). `E` means the earlier operation is authoritative; `L` means
the later operation is authoritative.

The one-byte codes are deliberately equal length, so changing authority does
not change operation, operand or decision-token positions within a pair. These
are a controlled authority markers, not claims about natural-language
instruction hierarchy.

## Training and evaluation

- Existing 22M dense backbone and tokenizer.
- Seeds 42 and 1337.
- 500 optimizer updates, effective batch size 12.
- Same pair schedule and target structure across conditions.
- D0/D1 use the existing operand and operation pointer heads.
- D2 adds only a decision-operation head reading the final decision token;
  operand pointers remain neural and are trained identically.
- No sealed-test access.

The deterministic oracle is evaluated for every condition. Training and
validation competence are reported separately. Counterfactual paired
correctness is primary: both authority variants must select their respective
operations correctly for the same operands and order.

## Required invariants

- Training/validation pair IDs are disjoint.
- Every pair has exactly four rows per condition: two operation orders and two
  authority codes.
- Within each counterfactual pair, operand and operation positions are
  identical, the authority position is identical, and the target operation
  flips.
- D2's decision token is after the authority field and all operation/operand
  content.
- D0/D1/D2 use identical model update counts, seeds and pair schedule.
- Checkpoint hashes, prediction counts and oracle outputs are verified.

## Interpretation

D1 success would show that making authority available before candidate
representations is sufficient for conditional pointer selection. D2 success
would show that a late authority cue can be used by a post-context decision
readout. Accuracy without counterfactual flips is not authority use.

Failure of D1 or D2 after passing training competence would motivate a binding
or optimization investigation. Failure before training competence is not
evidence against the corresponding decision interface.
