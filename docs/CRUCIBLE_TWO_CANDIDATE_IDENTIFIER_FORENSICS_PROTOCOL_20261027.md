# Crucible Two-Candidate Identifier Forensics v1

Date: 2026-10-27
Status: post-hoc analysis of committed prediction matrices
Branch: `research/two-candidate-forensics-20261027`

## Objective

Determine whether the matched-context probe errors are concentrated by ordered
candidate combination, target slot, candidate order or source-context stratum.
This analysis uses existing prediction matrices only. It performs no model
inference, training or checkpoint modification.

## Inputs

Use the committed prediction matrices from
`crucible_two_candidate_matched_context_probe_20261027_v1` for seeds 42 and
1337. The matrices contain 480 rows per seed: 240 familiar-regime rows and 240
recombined-regime rows, covering both source-context strata.

The analysis does not regenerate records, labels or splits. Familiarity is
defined by the candidate sets used by the original two-candidate training
population; recombination is defined by the candidate sets used by the
original validation population.

## Diagnostics

Report, with denominators:

1. Accuracy and errors for every ordered candidate pair.
2. Accuracy and errors for every unordered pair after collapsing reversals.
3. Target-slot accuracy and target/prediction confusion.
4. Error identities: expected candidate identifier versus predicted candidate
   identifier.
5. Source-context-stratum breakdown.
6. Reversal consistency for the same authority and context: whether the
   predicted identifier remains stable and whether the predicted slot flips
   when candidate order is reversed.
7. Abstention correctness and any out-of-range predictions.

## Interpretation safeguards

- Do not treat repeated rows from the same checkpoint as independent models.
- Do not call a pair-specific pattern proof of memorization or an embedding
  mechanism.
- Distinguish a fixed-slot shortcut from a fixed-identifier preference under
  reversal.
- Keep the familiar/regime and recombined/regime populations explicit.
- Do not convert this post-hoc analysis into a new validation score.

## Decision rule

- Persistent errors on the same unordered pairs across seeds and both context
  strata support a pair-specific behavioral weakness under this distribution.
- Errors that reverse with candidate order but preserve a preferred identifier
  are more consistent with candidate-identity selection than a simple fixed-slot
  shortcut.
- Balanced target-slot errors with perfect abstention weaken a broad output or
  abstention explanation.
- If these diagnostics do not distinguish exact-pair association from a more
  general but imperfect matching rule, recommend a preregistered balanced
  pair-coverage retraining experiment; do not run it in this analysis.

## Boundary

The analysis is descriptive and checkpoint-bound. It does not establish a
neural mechanism, an architectural limit, or transfer beyond the tested
candidate identities and source contexts.
