# Crucible Authority-Aware Role Binding Gate v1 — corrected distractor protocol

Date: 2026-10-25  
Status: preregistered CPU-only corrected rerun

This is the corrected rerun of the authority-aware gate. The preliminary v1
run is preserved separately because its familiar training layouts did not
contain explicit operand-distractor fields.

## Research question

Can explicit pointer supervision teach the existing parser to distinguish an
authoritative operation cue from a plausible competing cue while also
binding operands in the presence of numeric distractors?

## Frozen reference and arms

Use the accepted explicit role-supervision v3 parser/checkpoints as A0 and
reproduce its recorded results before new training. Compare:

- A0: frozen v3 pointer parser;
- A1: authority examples with explicit operand distractors, tuple loss only;
- A2: the same examples with tuple plus A/B/operation pointer losses;
- A3: the same pointer supervision with a position-balanced schedule;
- E3: oracle binding plus deterministic execution.

## Corrected training data

Every familiar training example contains:

- two distinct FIRST/SECOND operation cues;
- exactly one `authority=EARLIER` or `authority=LATER` label;
- explicit `noise_digit`, `decoy_a` and `decoy_b` fields using the same digit
  vocabulary as the target operands;
- balanced authoritative operation identity, authority side, operand roles and
  three familiar layouts.

The authoritative operation position is the earlier or later cue selected by
the authority label. No fixed recency rule is valid across the balanced
training set.

## Evaluation

Validation uses disjoint operand records and includes familiar layouts plus
reversed authority placement, new distractor arrangements, a nearest-decoy
stress layout and a novel conflicting-cue layout. The nearest-decoy layout is
reported separately because its stress subset intentionally fixes the
authority side; it is not treated as balanced evidence of authority reversal.

Report A position, B position, authoritative-operation position, full pointer
tuple and paired final-answer correctness separately. Require training
competence before interpreting held-out behavior.

## Invariants

- Same 22M backbone, tokenizer, optimizer family, 500 updates, effective batch
  size 12, seeds 42 and 1337, and canonical/factorized interfaces.
- A1/A2 use the existing B0 diverse/shuffled schedule; A3 uses one example
  from every authority-side × operation × familiar-layout stratum per batch.
- No executor retraining, persistent memory, recursion, architecture search or
  sealed-test access.
- All inputs must fit the existing context window and train/validation base
  record IDs must remain disjoint.

## Primary criterion

At least 90% paired correctness on held-out conflicting-operation examples,
sustained across authority positions and layouts. Success on one template is
not sufficient evidence of general authority reasoning.
