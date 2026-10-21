# C1b diagnostic benchmark proposal

Date: 2026-10-21
Status: proposal only; not executed
Execution boundary: CPU-only if approved; no model training, GPU, cloud or checkpoint download

## Purpose

C1 showed that the current synthetic environment strongly rewards a perfect,
cheap, immediately available verifier. C1b is designed to determine whether
that result is a property of the verification regime or a general policy
ordering result. It must be reviewed before execution and must not modify the
sealed C0/C1 artifacts.

The benchmark is diagnostic, not a learned-controller experiment. It tests
whether a bounded adaptive policy can choose among reasoning operations when
verification is not always the dominant action.

## Frozen inputs and provenance

- Governing controller commit: `6912a4a420d728a6916c310c0d5f6ef28dcf23ac`.
- C1 source traces remain immutable under
  `experiments/crucible_recursive_controller_c0_c1_20261021/`.
- C1b must use a newly generated fixture manifest with an independent seed,
  not the existing C1 test outcomes for policy selection or threshold tuning.
- The evaluator may retain exact latent answers for scoring only. Ordinary
  policies may observe only the candidate, declared tool contracts and tool
  responses.
- The C1b manifest must include clean, wrong, contradiction, information-
  deficient and shortcut-resistant cases, balanced across the four existing
  task families where the condition is applicable.

## Policies

Run all six policies under identical maximum budgets:

1. `one_pass`: answer and stop.
2. `fixed_depth_2`: critique, counterexample and revision, then stop.
3. `confidence_stop`: use the preregistered confidence threshold; never infer
   calibration from the evaluation outcomes.
4. `verification_only`: answer, call the verifier once, revise if the result
   rejects the answer.
5. `adaptive`: use the bounded controller's declared state, consistency,
   evidence and cost rules. An unavailable information-gain estimate remains
   unavailable; it must not be filled in from confidence.
6. `verification_first_adaptive`: call verification when the tool contract
   declares it reliable and affordable; when it is pending, noisy or
   unavailable, use a preregistered fallback sequence and re-evaluate after
   new evidence. This is the simple adaptive baseline the current C1 lacked.

No policy may inspect a latent answer, condition label, future tool response or
another policy's result. The condition's declared tool contract is observable
because real systems must know whether a tool is available and what it costs.

## Common budgets and cost accounting

Every policy receives the same ceilings:

```text
max depth: 8
max actions: 8
max evidence calls: 2
max counterexample calls: 2
max memory writes: 2
max estimated cost: 12 units
```

The C1 cost schedule remains fixed for comparable operations: answer 1,
critique 1, counterexample 1.5, evidence 2, verification 3 and revision 1.
C1b adds condition-specific verifier latency/reliability to the declared cost
without changing the common action ceiling. Report both realized cost and
budget utilization; equal ceilings must not be described as equal realized
cost.

## Preregistered condition cells

Each cell uses the same candidate distribution and independent fixture seed.
Tool noise and delayed results are generated from a seed recorded in the
manifest. The exact rates below are design parameters, not measured results.

| Cell | Verification contract | Intended diagnostic question |
| --- | --- | --- |
| R: reliable/cheap | exact, immediate, cost 3 | Does verification-first correctly dominate when it should? |
| I: imperfect | independent false-accept 10%, false-reject 10%, cost 3 | Can additional evidence reduce trust in a noisy verifier? |
| E: expensive | exact, immediate, cost 8 | Does the controller avoid verification when its opportunity cost is high? |
| D: delayed | exact result returns after one bounded transition, cost 4 | Can the controller preserve state and resume rather than hallucinate completion? |
| U: unavailable | verifier explicitly returns unavailable, cost 0.5 | Can evidence/counterexample paths replace verification safely? |
| A: acquisition-required | verifier unavailable; evidence reveals a missing fact with declared reliability/cost | Does information acquisition improve the answer before revision? |

The imperfect verifier's false outcomes are evaluator-labelled and independent
of policy selection. Memory commits require an independently verified result or
the preregistered evidence contract; a noisy self-report cannot authorize a
fact commit. Delayed and unavailable tools must produce explicit unresolved
states when the budget is exhausted.

## Metrics

Report by policy, condition, task family and seed:

- exact accuracy and initial accuracy;
- correction success and false-correction rate;
- unresolved and budget-exhaustion rates;
- verifier calls, evidence calls, counterexample calls and revisions;
- realized cost units, wall time and cost per resolved case;
- confidence Brier score, calibration error and selective risk;
- valid, rejected and rolled-back memory proposals;
- contradiction count and cycle-rejection count;
- delayed-tool resume integrity and provenance validity.

The primary efficiency comparison is the paired accuracy/cost frontier. A
policy is not credited for an accuracy gain that exceeds the common budget or
for a cheaper result that has a materially worse error rate.

## Decision rules

These rules are fixed before C1b execution:

1. The contract gate fails if any policy violates a budget, commits an
   unverified/contradictory memory record, loses a delayed transition, or
   produces a duplicate/missing committed result.
2. Adaptive advantage is **not established** merely by beating one-pass. It
   must be compared with `verification_first_adaptive` and the best simple
   policy in each cell.
3. A policy is on the useful frontier in a cell only when it has no lower
   accuracy and no higher mean realized cost than the comparison policy, with
   at least one strict improvement. Ties are reported as ties.
4. The adaptive hypothesis is considered supported only if the adaptive policy
   is on the useful frontier in at least four of six cells, has no safety
   regression, and the result survives the preregistered held-out seed set.
5. If verification-first is on the frontier in every cell, retain it as the
   default and do not add recursive utility complexity. If no policy dominates,
   preserve the Pareto set rather than selecting by narrative preference.

These are engineering decision rules for this diagnostic, not claims about
general intelligence or future competence.

## Shortcut and leakage controls

- Generate a new manifest seed; do not tune on C1 traces.
- Include whitespace, prompt-length and answer-distribution counterfactuals.
- Keep latent answers and condition outcomes outside ordinary controller state.
- Verify action traces are invariant to irrelevant prompt formatting.
- Audit whether any policy action depends on record ID, prompt length or target
  distribution.
- Preserve all failed hypotheses and raw traces.

## Expected artifact layout

If approved, write only to a new directory:

```text
experiments/crucible_recursive_controller_c1b_20261021/
  PROTOCOL.md
  MANIFEST.json
  RUN_CONFIG.json
  RESULTS_SUMMARY.md
  RESULTS.json
  TRACES.jsonl
  HASHES.json
```

No C1b results or hashes exist yet. This file is the preregistered proposal;
execution requires explicit review.
