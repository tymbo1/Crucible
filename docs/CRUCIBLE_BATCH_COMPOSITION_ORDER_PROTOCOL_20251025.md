# Crucible Batch-Composition and Update-Ordering Control

Date: 2026-10-25  
Status: preregistered CPU-only comparison

## Research question

Does the collapse observed under grouped positional exposure arise from batch
composition, optimizer-update ordering, or both?

## Arms

| Arm | Batch composition | Batch/update ordering |
|---|---|---|
| B0 | Diverse: 12 distinct operand-pair groups per batch | Shuffled batch order |
| B1 | Diverse: 12 distinct operand-pair groups per batch | Grouped/phase-ordered batch order |
| B2 | Same-pair: two complete operand-pair groups per batch | Shuffled batch order |
| B3 | Same-pair: two complete operand-pair groups per batch | Grouped batch order |

Each operand-pair group contains six records: `SELECT_FIRST` and
`SELECT_SECOND`, each in the three trained layouts from the paired-positional
exposure experiment. Every expanded training record is presented exactly 25
times in every arm. Each arm therefore has 500 updates, effective batch size
12, and 6,000 record presentations. The schedules are fixed before training
and shared across seeds and interfaces.

## Fixed controls

- Existing 22M dense-control model, tokenizer, evaluator and optimizer.
- Canonical and factorized interfaces.
- Seeds 42 and 1337.
- Learning rate, weight decay, gradient clipping and 500-update budget from
  the paired-positional exposure control.
- No architecture change, consistency loss, memory, recursion or symbolic
  execution assistance.
- Existing immutable train/validation pair splits and the sealed test boundary.

## Measurements

For every update, record task loss, token correctness, gradient norm,
full-gradient cosine similarity to the preceding update, unique operand-pair
groups, layout diversity, operation diversity and record exposure counts.

For every checkpoint, evaluate all trained layouts and the two previously
unseen layouts on both training and validation pairs. Report ordinary exact
accuracy and paired conditional correctness separately.

The primary comparison is B0 versus B1 versus B2 versus B3 under matched
record exposure. A0/A1 from the preceding experiment are historical context,
not arms in this protocol.

## Planned interpretation boundaries

- If B0 and B1 differ while their batches are identical, update ordering is a
  supported contributor.
- If B2 and B3 differ while their batches are identical, update ordering also
  affects same-pair exposure.
- If diverse and same-pair arms differ under matched ordering, batch
  composition is a supported contributor.
- A training-fit difference is an optimization result, not evidence of
  generalization.
- Validation results are interpreted only after reporting training competence.
- No internal mechanism such as gradient interference is claimed from
  gradient cosine alone.

## Checkpoint-only reference evaluation

The prior successful A0 shuffled-exposure checkpoints are evaluated read-only
across all layouts. This does not alter or retrain those artifacts and is
reported separately from the new B0-B3 arms.

## Boundaries

The sealed test set is not opened. Existing results are preserved. If any
invariant fails, the run stops rather than silently changing the data,
optimizer, architecture or budget.
