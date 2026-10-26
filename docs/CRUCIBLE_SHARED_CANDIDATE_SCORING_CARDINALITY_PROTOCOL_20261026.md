# Crucible Shared Candidate-Scoring Cardinality Gate v1

Date: 2026-10-26  
Status: preregistered CPU diagnostic

## Objective

Separate a position-based fixed-slot readout from a shared candidate-scoring
readout on an explicit matching task. Training uses two and three candidates;
transfer uses four, five and eight candidates without retraining.

Each candidate has a numeric identifier. The late authority query contains an
identifier. The correct decision is the candidate whose identifier matches the
query, or abstention when no candidate matches. Candidate order is varied so
that position and identifier are separable.

## Arms

- `SLOT_HEAD`: a fixed output head over eight candidate slots plus abstention,
  with invalid slots masked for each example;
- `SHARED_SCORER`: one query/key scoring function applied to every candidate
  representation plus a shared abstention logit;
- deterministic identifier-match oracle: positive control for the data and
  evaluator, not a learned result.

The dense backbone, tokenizer, optimizer, update count, batch size, interfaces
and seeds are matched between learned arms. The readout is the only learned
component changed between arms. No sealed test set is used.

## Partitions

- Training: candidate counts 2 and 3, compact layout.
- Validation-2/3: candidate counts 2 and 3, held-out candidate sets and a
  padded/reordered layout.
- Transfer-4/5/8: candidate counts 4, 5 and 8, held-out layouts and candidate
  permutations. Candidate identifiers remain in the training vocabulary.

Each partition includes counterfactual pairs with the same candidates and
layout but different present authority identifiers. Absent identifiers require
abstention.

## Metrics and gates

Report training pointer accuracy, same-count validation accuracy, each transfer
cardinality, abstention accuracy and counterfactual flips/two-way correctness.
The 95% training pointer gate is required before interpreting learned transfer
results. The primary cardinality-transfer metric is paired correctness, not
row-level accuracy alone.

## Interpretation boundaries

Success by `SHARED_SCORER` would support a candidate-independent matching bias
on this synthetic identifier task. Failure would not prove that shared scoring
is impossible; oracle success with learned failure would localize the problem
to neural representation/extraction or optimization. Neither outcome
establishes general reasoning or autonomous learning.
