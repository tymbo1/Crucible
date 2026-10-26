# Crucible Rule-Induction Benchmark Protocol

Date: 2026-10-24  
Status: benchmark specification frozen before generation  
Execution: CPU-only generation and audit; no model training

## Objective

Test whether a model can infer an unfamiliar rule from demonstrations and apply it to a new input, rather than memorize fixed input/output pairs.

Each task hides a modulo-ten rule `R`, presents 1, 2, 4 or 8 compact demonstrations, and asks for the output on a ninth input. The same task supports all four demonstration counts, so performance can be measured as evidence accumulates. Prompts are audited against the dense control's 320-token context window before any model run.

## Rule families

The initial finite hypothesis library contains:

- two-input affine rules: `(a*x + b*z + c) mod 10`;
- shifted maximum rules: `(max(x,z) + c) mod 10`;
- shifted absolute-difference rules: `(abs(x-z) + c) mod 10`;
- threshold-select rules with a threshold and separate low/high offsets.

Behaviorally duplicate rules are deduplicated before splitting. Candidate consistency and query-identifiability are computed against the resulting library. The hidden family and parameters are never included in the task prompt.

## Splits and leakage controls

Rule instances, not individual demonstrations, are assigned to train, validation or test using a family-stratified split. Each family contributes at least two validation and two test instances. A rule-instance identifier may occur in only one split. Input pairs can recur across rules because that is part of the induction problem; hidden rule identity and parameters cannot.

For every task:

- demonstrations are unique input pairs;
- the query input is not in the demonstrations;
- the oracle is independently recomputed;
- candidate rules consistent with each demonstration prefix are counted;
- whether all consistent candidates agree on the query is recorded.

Tasks with unresolved ambiguity are retained and labelled rather than silently removed. Accuracy must later be reported separately for identifiable and non-identifiable cases.

## Evaluation conditions

The benchmark exposes nested prompts with 1, 2, 4 and 8 demonstrations. The primary learning curve is exact query accuracy by demonstration count. Secondary metrics are:

- accuracy conditional on query-identifiable tasks;
- accuracy on ambiguous tasks;
- exact-rule identification rate;
- rule-family and split performance;
- output calibration and abstention, if supported by the model;
- whether additional demonstrations improve the same task monotonically.

No fixed accuracy threshold is claimed for rule induction until the benchmark audit establishes its difficulty and identifiability profile.

## Research controls

The first model experiment, if approved after audit, should compare:

1. an ordinary prompt-trained control;
2. a representation-matched control with the same model, optimizer and token budget;
3. a deterministic candidate-search oracle as an upper bound, reported separately from learned performance.

Training and evaluation must use disjoint rule instances. Test results are not used for checkpoint selection or curriculum design.
