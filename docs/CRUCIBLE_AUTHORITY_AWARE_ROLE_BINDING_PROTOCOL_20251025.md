# Crucible Authority-Aware Role Binding Gate v1

Date: 2026-10-25  
Status: preregistered CPU-only intervention

## Research question

Can the supervised pointer parser distinguish an authoritative operation cue
from a plausible competing cue and transfer that selection to unseen operand
pairs and layouts?

## Frozen reference

The accepted explicit role-supervision run is
`experiments/crucible_explicit_role_supervision_20251025_v3`. Its four
interface/seed checkpoints, predictions and hashes remain unchanged. The
authority run must reproduce the v3 pointer results from the recorded
artifacts before interpreting new results.

## Conditions

| Arm | Parser/training | Execution |
|---|---|---|
| A0 | Frozen v3 pointer parser | Deterministic pointer-decoded executor |
| A1 | Same parser heads; authority examples; tuple loss only | Deterministic classifier-decoded executor |
| A2 | Same parser heads; authority examples; tuple plus pointer losses | Deterministic pointer-decoded executor |
| A3 | Same as A2 with a position-balanced authority schedule | Deterministic pointer-decoded executor |
| E3 | Oracle binding | Deterministic executor |

A1 is the no-explicit-pointer-supervision control. A2 and A3 receive the same
authority-labelled examples and total update budget; A3 changes only the
deterministic batch schedule so each update balances authority side, operation
identity and familiar layout strata.

## Authority data

Every authority example contains two distinct operation cues, one FIRST and
one SECOND, plus exactly one `authority=EARLIER` or `authority=LATER` label.
The target operation is the cue selected by that label. Authority is balanced
between earlier and later cues, and operation identity, operand roles and
familiar layouts are audited before training.

Training uses three familiar layouts with 240 records per interface. The
validation source is disjoint and contains unseen operand records. Evaluation
also includes:

- a reversed authority-position layout;
- digit and operand distractors using the same digit vocabulary;
- a layout where a non-authoritative operation is nearest the answer boundary;
- a conflicting-cue layout with the authority relation in a novel position.

The parser is separately scored for A position, B position, authoritative
operation position and the complete bound tuple. Paired final-answer
correctness is the primary execution metric.

## Training invariants

- Same 22M dense backbone, tokenizer, optimizer family, 500 updates, batch 12,
  seeds 42 and 1337, and canonical/factorized interfaces as v3.
- No executor retraining, persistent memory, recursion, architecture search or
  sealed-test access.
- A1/A2 use the existing B0 diverse/shuffled schedule. A3 uses a deterministic
  schedule with one example from each authority-side × operation × familiar
  layout stratum per batch.
- Any record exceeding context length, missing role spans or unbalanced target
  strata stops the run before training.

## Success criterion

The primary target is at least 90% paired correctness on held-out conflicting-
operation examples, sustained across authority positions and layouts. Success
on one template alone is not sufficient evidence of general authority
reasoning.

## Required artifacts

Protocol manifest, immutable authority datasets, schedules, per-arm
checkpoints and hashes, per-record parser and output predictions, training
traces, baseline-reproduction audit, summary, final report and scientific
interpretation.
