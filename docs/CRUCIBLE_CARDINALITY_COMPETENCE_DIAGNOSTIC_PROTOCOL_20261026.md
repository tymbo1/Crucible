# Crucible Cardinality Competence Diagnostic v1

Date: 2026-10-26  
Status: frozen before execution  
Scope: CPU-only training-competence diagnostic; no transfer evaluation

## Objective

Determine whether the explicit-ID cardinality training failure is more
consistent with the current dense representation or with the deterministic
cyclic training schedule. This is a competence diagnostic only. It does not
test transfer to four, five or eight candidates and does not reopen any sealed
test set.

## Novelty audit

Before execution, the repository ledger was searched for `cardinality`,
`representation`, `schedule` and `competence`. The search found the prior
explicit-ID cardinality failure, the earlier shared-scorer transfer failure,
the batch-composition study, the position-generalization study and related
role/parser controls. No prior experiment tested this exact two-factor
representation-by-schedule competence matrix on the frozen cardinality train
records. The prior cardinality experiment remains the control; this run must
not be pooled with its transfer results.

## Frozen source and scope

- Source data: `experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/data/canonical/train.jsonl`.
- One interface: `canonical`.
- One initialization seed: `42`.
- 576 training rows, unchanged from the accepted cardinality experiment.
- 500 updates, effective batch 12, 6,000 row presentations.
- Optimizer family, learning rate, weight decay and gradient clipping match the
  existing runner where the cell uses the dense model.
- Training competence threshold: at least 95% row-level pointer accuracy on
  all 576 training rows.
- No validation, transfer, candidate-count generalization or sealed-test
  claim is made by this diagnostic.

## Four cells

| Cell | Representation | Schedule | Purpose |
|---|---|---|---|
| D0 | Existing dense hidden states and shared scorer | Existing deterministic cyclic schedule | Reproduce the relevant failed training condition |
| D1 | Existing dense hidden states and shared scorer | Deterministic shuffled/interleaved schedule | Test schedule/order contribution |
| D2 | Explicit candidate-ID/query/slot/count features with a shared scorer | Existing deterministic cyclic schedule | Test whether structured features remove the representation burden |
| D3 | Explicit candidate-ID/query/slot/count features with a shared scorer | Deterministic shuffled/interleaved schedule | Test the representation/schedule interaction |

The structured cells use candidate IDs and query IDs as one-hot features, plus
candidate slot and candidate-count features. They do not receive the target
slot, equality labels or any privileged answer. Candidate spans and labels
remain the same deterministic source metadata used by the existing experiment.

## Metrics and controls

Report row-level pointer accuracy, abstention accuracy, counterfactual-pair
correctness on the training rows, loss curves, gradient norms, parameter
counts, presentation counts and exact source-data hashes. The primary gate is
training pointer accuracy; paired correctness is secondary and descriptive.

The deterministic identifier rule is recomputed as a positive control. Any
source-data, label, span, count or schedule invariant failure stops the run.
No checkpoint from the earlier experiment is modified.

## Interpretation rules

- High D0/D1 and low D2/D3 would implicate the structured feature
  implementation or its optimization, not the prior dense representation.
- Low D0/high D1 would support a schedule contribution under this budget.
- High D2/D3 with low D0/D1 would show that explicit structure makes the
  competence task learnable, without proving the dense architecture cannot
  learn it.
- If all cells fail, the bottleneck remains unresolved; do not infer an
  architectural ceiling.
- No cell result authorizes cardinality transfer claims unless the training
  competence gate is passed first.
