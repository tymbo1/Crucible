# Scientific Interpretation

This is a post-hoc analysis of the committed matched-context prediction
matrices. It did not run the model again. No checkpoints, labels or source
matrices were modified.

## Directly supported patterns

- Familiar candidate sets were perfect for both seeds across both source-context
  strata.
- Recombined errors are concentrated by unordered candidate family. The `(2,5)`
  and `(3,6)` families fail persistently in both seeds; `(0,3)` is perfect in
  both; `(1,4)` is seed-dependent.
- Every recombined abstention row is correct. Non-abstention errors select the
  other candidate rather than producing a new output class.
- Target-slot performance is not a stable one-sided shortcut: seed 42 has a
  modest slot-0/slot-1 difference, while seed 1337 is nearly balanced.
- Candidate-order reversal usually preserves the predicted identifier and
  flips the predicted slot. This weakens a simple fixed-slot explanation. In
  the persistent error families, the model often preserves the same *wrong*
  candidate identity across reversal.

## What remains unresolved

These data support a pair-specific behavioral weakness, but cannot distinguish
exact pair association from an imperfect identifier-matching rule. Only four
unseen unordered recombined pairs are available, and the analysis contains two
checkpoints rather than independent model populations. No neural mechanism or
architectural conclusion is justified.

## Single next controlled experiment

Run a preregistered balanced pair-coverage retraining experiment, keeping the
22M dense `SHARED_SCORER`, cyclic schedule, compact layout, optimizer budget and
competence gate fixed. Rotate held-out unordered candidate pairs across several
balanced splits while ensuring every identifier, target slot, candidate order
and source context is represented in training. Evaluate both held-out pair
orders with counterfactual authority groups.

If errors move with the held-out pairs, exact pair coverage is implicated. If
the same identifier families fail across rotated holdouts, an identifier- or
representation-specific effect is more plausible. This experiment should be
run only after a repository novelty audit; it was not run here.
