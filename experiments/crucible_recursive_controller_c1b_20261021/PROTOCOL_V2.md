# C1b revised two-track protocol

Date: 2026-10-21
Status: proposal only; supersedes the single-track execution plan in
`PROTOCOL.md` for review
Execution boundary: CPU-only if approved; no GPU, cloud, checkpoint download or Crucible model training

## Research question

C1 established the controller contracts but used a regime in which an exact,
immediate verifier was cheap and always available. C1b separates two questions:

- **C1b-A:** Do the frozen policies behave sensibly when the environment changes?
- **C1b-B:** Can a small CPU learner discover better strategy selection from
  measured outcomes, including after an environment shift?

Neither track is allowed to modify the sealed C0/C1 results or retune the
existing adaptive policy against them.

## Provenance and frozen inputs

- Sealed baseline commit: `6912a4a420d728a6916c310c0d5f6ef28dcf23ac`.
- Sealed C1 trace SHA-256:
  `b325c142f734523d088e1ab8533f45cc17fcd7fbce10d31b40921a48a8d36e98`.
- The existing C1 directory remains immutable.
- C1b uses a newly generated fixture manifest with an independent seed and
  independent policy-selection/evaluation seeds.
- Latent answers are evaluator-only. A decision-time policy sees only its
  current state, declared tool capabilities, observed tool responses and
  inherited budgets.
- C1b-B training episodes and evaluation episodes use disjoint task IDs,
  environment seeds and condition schedules. No evaluation outcome selects a
  policy, coefficient or stopping threshold.

## Track A — frozen-policy environment diagnostics

### A1. Policies

Run these six policies without changing their decision rules:

1. one-pass;
2. fixed-depth;
3. confidence-stop;
4. verification-only;
5. existing heuristic adaptive;
6. verification-first adaptive.

The sixth policy is a simple control: call verification when the declared
contract says it is reliable and affordable; on pending, noisy or unavailable
results, use a fixed fallback sequence and reassess after new evidence. It is
not allowed to learn from the evaluation set.

### A2. Environment cells

Use the six cells from the earlier proposal, with the exact contracts frozen
before execution:

| Cell | Contract |
| --- | --- |
| R | exact immediate verifier, cost 3 |
| I | independent 10% false-accept and 10% false-reject verifier, cost 3 |
| E | exact immediate verifier, cost 8 |
| D | exact result delayed by one bounded transition, cost 4 |
| U | verifier unavailable and returns an explicit failure, cost 0.5 |
| A | verifier unavailable; evidence acquisition reveals a missing fact with declared reliability and cost |

Add contradiction frequency and evidence availability as orthogonal manifest
fields, rather than creating special cases after inspecting outcomes. Include
tight total-cost ceilings of 4, 6, 8, 10 and 12 units as separate preregistered
cost-frontier slices. A run must report both the common ceiling and its actual
realized cost.

### A3. Track-A outputs

Report accuracy, correction and false-correction rates, unresolved states,
tool calls, realized cost, cost per resolved case, calibration, memory validity,
contradiction handling, cycle handling and delayed-transition recovery by
policy, cell, family and seed.

The purpose of A is diagnosis. It passes even if verification-first wins every
cell, provided all safety contracts hold and the outcome is reported as a
valid negative result for more complex control.

## Track B — learned strategy-selection feasibility

Track B is a CPU learning experiment over the same bounded action interface. It
does not train Crucible, a language model or a neural architecture.

### B1. State and action contract

The learner observes only:

- remaining depth, action, tool and cost budgets;
- candidate confidence marked as uncalibrated until independently calibrated;
- observed evidence reliability and provenance status;
- verifier outcomes already received, including pending/unavailable status;
- contradiction count, unresolved-claim count and prior action outcomes;
- declared tool cost/reliability fields when the environment exposes them;
- elapsed bounded steps and memory-write queue state.

It must not observe record IDs, latent answers, future tool outcomes, test-set
labels or another policy's result. Actions are answer, verify, critique,
counterexample, evidence search, revise and stop. Invalid actions are masked by
the same contract engine used in C0/C1.

### B2. Feedback and reward

The learner receives outcome feedback only after an action or episode step.
The initial CPU feasibility reward is:

\[
r_t = \Delta Q_t + \lambda E_t + \rho C_t
      - \kappa Cost_t - \mu Error_t,
\]

where \(\Delta Q_t\) is verified progress, \(E_t\) is independently measured
useful evidence, \(C_t\) is consistency/provenance status and the cost/error
penalties are in declared normalized units. Unverified self-confidence is not
reward. The future-competence term is disabled in C1b; it is a separate later
hypothesis requiring an untouched future-task protocol.

### B3. Learning baselines

Implement and compare in this order:

1. **Contextual-bandit baseline:** fixed-feature linear action-value estimates
   with a preregistered exploration schedule. It learns immediate action value
   but does not bootstrap through delayed transitions.
2. **Rule-based controls:** verification-first adaptive and the frozen
   heuristic adaptive policy.
3. **Bounded sequential controller:** only if delayed evidence or multi-step
   state transitions create a measurable residual after the bandit. Use a
   tabular/linear Q estimator with fixed discount, finite horizon and explicit
   exploration cap before considering any neural approximator.

The sequential controller must earn a go decision from the bandit comparison;
reinforcement learning is not run merely because it is more elaborate.

### B4. Environment-shift test

Use two shift modes:

- **Declared shift:** the tool contract exposes changed cost/reliability. This
  tests conditional strategy selection.
- **Inferred shift:** the visible contract remains unchanged while observed
  verifier outcomes change according to a preregistered schedule. This tests
  adaptation from experience rather than reading a regime label.

Train on disjoint episodes from a subset of cells. Evaluate on held-out task
families, held-out seeds and at least two unseen condition transitions. Measure
adaptation delay, cumulative regret against an oracle selector available only
to the evaluator, post-shift accuracy, false corrections and cost.

## Common safety and evaluation rules

- All policies use the same maximum depth, action, tool, memory and cost
  ceilings as C0/C1.
- Report realized costs; do not call unequal-cost runs matched-cost evidence.
- Exact or noisy verifier outcomes are tool feedback, never hidden decision-time
  labels.
- Memory commits require provenance, verification or the explicitly declared
  evidence contract, and no unresolved contradiction.
- Delayed results must be resumable and cannot be replaced by guessed facts.
- Cycle detection, rollback, budget exhaustion and duplicate/missing packet
  checks remain hard gates.
- Confidence calibration is measured on independent calibration data and is
  never tuned on held-out evaluation outcomes.
- Preserve raw traces, rejected actions and failed hypotheses.

## Metrics and go/no-go gates

### Gate A — environment and contract validity

Go only if all six cells complete with zero contract violations, no invalid
commits, no lost delayed transitions and reproducible traces. A policy winning
is not required.

### Gate B — contextual-bandit feasibility

Go only if the bandit beats the best frozen rule on at least one held-out
environment shift without a safety regression, and does not lose more than one
percentage point of accuracy in any other held-out cell. Otherwise retain the
best deterministic rule and stop the learned-policy track.

### Gate C — sequential controller

Run only after Gate B and only if delayed/persistent state produces a measured
bandit limitation. The bounded Q controller must lie on the accuracy/cost
Pareto frontier against verification-first adaptive in at least four of six
cells, including an inferred-shift cell, with no safety regression. Otherwise
stop at the simpler bandit or deterministic policy.

### Future competence

Do not claim future competence from C1b. A later experiment must intervene on a
training or memory process and compare an untouched future-task distribution
against a matched no-intervention control.

## Estimated CPU-only resources

These are planning estimates, not measurements. The sealed C1 replay took
approximately one second for 1,280 policy cases on the current PC.

| Work | Proposed scale | Estimated wall time | Peak RAM | Trace storage |
| --- | --- | ---: | ---: | ---: |
| C1b-A diagnostics | 6 cells × 6 policies × 8 seeds, 256 cases/seed | 2–10 minutes | <1 GB | 0.2–2 GB raw; <100 MB aggregates |
| Contextual bandit | 10k–50k CPU episodes, horizon ≤8 | 2–15 minutes | <1 GB | 50–500 MB |
| Bounded Q feasibility | only after bandit gate; ≤100k episodes | 5–30 minutes | <2 GB | 0.1–1 GB |

No GPU-hour or cloud-cost budget is requested. Estimates must be replaced by
measured CPU timing after approval and before increasing the episode count.

## Expected artifacts

Create a new versioned directory only after approval:

```text
experiments/crucible_recursive_controller_c1b_20261021/
  PROTOCOL.md              # original proposal, preserved
  PROTOCOL_V2.md           # this superseding two-track protocol
  MANIFEST.json
  RUN_CONFIG.json
  C1B_A_RESULTS.json
  C1B_B_RESULTS.json
  RESULTS_SUMMARY.md
  TRACES.jsonl
  HASHES.json
```

This protocol contains no C1b results. No C1b execution, learner training or
model/GPU/cloud work is authorized by this document.
