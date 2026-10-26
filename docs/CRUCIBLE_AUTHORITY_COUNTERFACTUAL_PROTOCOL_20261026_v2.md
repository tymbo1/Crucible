# Crucible Authority Counterfactual Diagnostic v2

Date: 2026-10-26  
Status: preregistered position-safe correction

## Purpose

The first counterfactual export used equal-width authority fields so every
layout had identical operation and operand positions. That introduced two
trailing spaces into the `LATER` field, a surface form absent from the v2
training data. Its results are retained as exploratory fixed-width evidence,
but are not sufficient to diagnose authority use in the existing checkpoints.

This corrected diagnostic uses the natural, unpadded `EARLIER` and `LATER`
labels and restricts the primary analysis to the existing
`UNSEEN_REVERSED_AUTHORITY` layout. In that layout, the authority field comes
after both operation cues and both operands, so changing the label leaves all
operand and operation positions unchanged without introducing padding.

## Frozen source and evaluation

Use only the v2 validation records and verified v2/v3 checkpoints. Evaluate
A0, A1, A2, A3 and E3 across canonical/factorized interfaces and seeds 42 and
1337. No retraining, new split, architecture change or sealed-test access.

For every validation record, create both operation orders (`FS` and `SF`) and
both raw authority labels. The target operation is the first cue for
`EARLIER` and the second cue for `LATER`. All other content is identical.

## Required invariants

- Raw authority labels are exactly `authority=EARLIER` and
  `authority=LATER`.
- Both operation positions and both operand positions are identical within
  every counterfactual pair.
- The target operation reverses exactly when the authority label reverses.
- Validation source IDs are disjoint from frozen v2 training IDs.
- Checkpoint hashes, prediction counts and deterministic execution verify.

## Primary measures

Report operation-pointer flips, correct two-way pointer flips, final-output
changes, correct two-way final-output flips on unequal operands, first/second
cue preference, operation preference, and breakdown by operation order,
interface and seed.

The position-safe result is the primary authority counterfactual evidence.
The earlier fixed-width full-layout export remains a separate exploratory
surface-normalization diagnostic and must not be pooled with this result.

