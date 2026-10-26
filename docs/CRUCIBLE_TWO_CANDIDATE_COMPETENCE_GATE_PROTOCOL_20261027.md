# Crucible Two-Candidate Competence Gate v1

Date: 2026-10-27  
Status: frozen before execution  
Scope: isolated two-candidate training competence, followed by gated validation

## Objective

Determine whether the dense cardinality matcher can learn both candidate slots
when the mixed-cardinality confound is removed. This is a competence repair,
not a candidate-count transfer experiment.

## Novelty and related controls

The research ledger contains related explicit-ID cardinality, shared-scoring,
two-candidate authority-selection and slot-readout experiments. It contains no
prior experiment that isolates the canonical two-candidate rows with balanced
slot-0, slot-1 and abstention classes under this training-competence gate.

The prior 576-row cardinality experiment remains the source and implementation
control. Its 3-candidate rows are excluded here; no new split is generated.

## Frozen data

- Training source: `experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/data/canonical/train.jsonl`.
- Validation source: `experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/data/canonical/validation.jsonl`.
- Keep only rows with exactly two candidates.
- Training: 144 rows, 48 target slot 0, 48 target slot 1 and 48 abstentions.
- Validation: 96 rows, 32 target slot 0, 32 target slot 1 and 32 abstentions.
- Preserve original record IDs, inputs, labels and counterfactual-pair IDs.
- No sealed test data and no transfer split.

## Arms

Train the existing dense `SHARED_SCORER` model with the existing optimizer,
learning rate, clipping and 22M-parameter architecture.

| Schedule | Seeds | Purpose |
|---|---|---|
| Deterministic cyclic | 42, 1337 | Reference ordering |
| Deterministic shuffled, seed 20261027 | 42, 1337 | Ordering control |

Use 500 updates, effective batch size 12 and the same total update budget as
the prior diagnostic. Each arm is independently initialized with its stated
seed.

## Gates and metrics

Primary training gate: at least 95% row-level pointer accuracy on all 144
training rows.

Report separately:

- slot-0, slot-1 and abstention accuracy;
- paired counterfactual correctness for the 48 training groups;
- validation slot and abstention accuracy only for arms passing the training
  gate;
- validation paired correctness for the 32 validation groups;
- loss and gradient traces, checkpoint hashes and exact source hashes.

If an arm fails the training gate, its validation result is recorded as not
interpretable and is not used as evidence of transfer.

## Invariants

- No architecture modification.
- No new generated split.
- No source-data or prior-checkpoint modification.
- No target, equality label or privileged answer feature.
- Deterministic candidate-span parsing and oracle verification must pass.
- Stop if any source, label, class-balance or split invariant fails.

## Interpretation boundary

A passing training gate establishes competence on the isolated two-candidate
distribution. Gated validation performance is evidence of transfer to the
existing two-candidate validation rows only. Neither result establishes
variable-cardinality generalization.
