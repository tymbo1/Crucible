# Crucible cognitive evaluation framework

Date: 2026-10-21  
Status: design-only; no training, checkpoint download, cloud rental or frozen-artifact modification authorized.

## Purpose

Crucible's existing NLL/BPB results measure language prediction. They do not,
by themselves, establish mathematical reasoning, abstraction, metacognition,
planning, creativity, autonomy or persistent learning.

This framework keeps language modelling as a necessary baseline while adding
separate, falsifiable evaluations for the capabilities that motivate the
larger ProjectBrain/Crucible programme.

The central rule is:

> A mechanism earns an architectural claim only when it improves a held-out
> capability over a simpler control under matched or explicitly accounted
> information, compute, memory and tool budgets.

## Current evidence boundary

The E4/E8/E16 matched-active-compute results provide a preliminary sparse-
capacity signal on the tested byte-level TinyStories-style task:

| Variant | Total parameters | Mean validation NLL |
|---|---:|---:|
| E4 | 43,151,104 | 0.792905 |
| E8 | 85,511,936 | 0.761320 |
| E16 | 170,233,600 | 0.749846 |

This supports the narrow statement that additional sparse capacity improved
language-modelling loss in that protocol. It does not establish functional
expert specialization or improvement on unfamiliar reasoning tasks.

The CPU-only artifact audit found metrics and initial seed anchors for the
expert-count runs, but no trained final E4/E8/E16 checkpoints or routing traces
that can support checkpoint-level causal expert ablations. Consequently,
leave-one-expert-out and expert-replacement tests cannot be run from the
existing package without a new state-saving experiment.

Relevant source records:

- `crucible_sparse_capacity_gate_v1_20261017/analysis_v1_20261017/`
- `crucible_expert_count_ab_v1_20261017/full_run/run_ledger.json`
- `crucible_telemetry_ab_100k_analysis_v1_20261016/specialization_analysis.json`
- `docs/CRUCIBLE_MOE_ARCHITECTURE_STATUS_AND_GATES_20261021.md`

## Evaluation dimensions

Each dimension needs its own task manifest, controls, primary metric and
failure analysis. A model's fluent explanation is not accepted as evidence of
the underlying capability.

| Dimension | What to measure | Required safeguards |
|---|---|---|
| Language competence | EOS-excluded byte NLL/BPB, held-out generalization and calibration | Match raw-byte exposure, context, tokenizer and optimizer updates |
| Mathematics | Exact arithmetic, algebraic transformations, unseen-rule induction and proof validity | Hold out values, compositions and rule combinations; score answers independently |
| Logic | Deduction, contradiction detection, consistency and proof checking | Include adversarial distractors and invalid-premise controls |
| Relational cognition | Delayed key-value retrieval, entity tracking, analogy, multi-hop composition and binding under interference | Hold out relation combinations and test length extrapolation |
| Metacognition | Error detection, confidence calibration, selective prediction and successful revision | Self-reported uncertainty alone does not count; use Brier/ECE, error AUROC and correction rate |
| Creativity | Novel, coherent and useful outputs satisfying explicit constraints | Separate novelty from randomness; use held-out constraints, similarity checks and blinded scoring |
| Planning | Subgoal quality, dependency handling, execution success, cost and replanning after failure | Require executable plans and measure outcomes, not plan prose |
| Autonomy | Goal persistence, bounded subgoal generation, progress monitoring, interruption recovery and completion | Separate goal quality from goal pursuit; record false progress and unauthorized actions |
| Continual learning | Retention, forward transfer, adaptation and forgetting after validated updates | Compare CMS2-only memory, weight updates, replay and no-update controls |

## Capability versus architecture

Every comparison must state which source of improvement it is testing.

### Capacity versus mechanism

A failure on a cognitive task is not an architectural failure unless the task
is learnable by an appropriate control at the tested scale and training
budget. Before rejecting a mechanism, establish:

1. a competence floor on the task family;
2. a simpler control that can learn the task under the same exposure;
3. sufficient optimization and inference budget;
4. whether increasing capacity changes the result without changing the
   mechanism;
5. whether the mechanism still contributes after the capacity comparison.

The inverse warning also applies: a larger model passing a task does not prove
that its architecture is responsible. Scale, curriculum, pretrained knowledge
and extra inference computation must be reported separately.

Crucible should use three research tiers:

| Tier | Intended use | Interpretation boundary |
|---|---|---|
| Approximately 130–170M | Routing, expert interventions, binding, retention, algorithmic generalization and small trained reasoning tasks | Cheap mechanism science; not the sole judge of advanced cognition |
| Approximately 1B | Richer reasoning, calibration, adaptive computation and learned planning | Medium-scale mechanism validation; pretrained or distilled foundations are separate ablations |
| Approximately 7–12B | Integrated recursive, symbolic, tool-using and goal-pursuit systems | Later integration tier; not the first architecture-discovery target |

These are planning tiers, not established capability thresholds. The same task
family should be evaluated across tiers only when the curriculum, tokenizer,
compute accounting and control geometry are documented.

Every MoE report must distinguish:

- total parameters, which affect storage and aggregate capacity;
- active parameters per token, which approximate conditional compute;
- training exposure per expert, which affects how much opportunity each expert
  has to learn;
- context, data diversity and training duration.

A sparse 170M-parameter model and a dense 170M-parameter model are not
automatically equivalent in any of these quantities.

### Capacity controls

At minimum, report separate comparisons for:

1. **Matched active compute:** same active expert work, different dormant
   capacity.
2. **Matched total parameters:** same parameter budget, different routing or
   allocation.
3. **Matched wall-clock or GPU budget:** same measured execution budget,
   different capability achieved.
4. **Dense control:** a simpler dense model with explicitly matched training
   bytes and active compute where feasible.

E4/E8/E16 must not be described as proving that experts are better than dense
   capacity until the appropriate dense and total-parameter controls exist.

For advanced cognitive claims, include a competent pretrained model as a
benchmark reference when practical. Its score establishes that the task and
feedback loop are usable by a capable language system; it does not establish
Crucible superiority. Teacher inference, tokenizer conversion and distillation
cost remain part of the comparison.

### Data and training controls

Record and match, or explicitly account for:

- raw training bytes and optimizer updates;
- task-family composition and example difficulty;
- context length and packing rules;
- tokenizer and vocabulary;
- initialization and pretrained status;
- optimizer, learning-rate schedule and precision;
- active and total parameters;
- tool calls, retrieval, memory reads and recursive passes;
- wall time, GPU time, peak VRAM and energy/cost where available.

Pretrained initialization is an enabling ablation, not a free architectural
control. A pretrained foundation must be compared with a from-scratch model
and its teacher, tokenizer conversion and distillation cost must be reported.

## Task curriculum and sufficiency

A failure on an unfamiliar task is ambiguous if the model was never trained on
the relevant task family. Evaluation therefore proceeds in stages.

### Stage A — deterministic task generators

Build CPU-testable generators and immutable manifests for arithmetic, symbolic
rules, delayed retrieval, relation composition, planning dependencies and
interruption/recovery. Split by underlying rule, composition, entity names,
surface form and length—not merely by random examples.

### Stage B — competence calibration

Before making an architectural claim, establish that each model has learned
the training task family above a declared competence floor. Report training,
validation and held-out-composition performance separately.

### Stage C — generalization evaluation

Use held-out rules, relation combinations, distractor patterns, longer
sequences and changed surface forms. Keep test generators and answers sealed
from the training process.

### Stage D — mechanism ablations

Only after the task is learnable should the experiment remove or add sparse
experts, persistent state, recursion, symbolic verification or adaptive
compute. One mechanism changes per gate.

## Expert specialization gate

The existing routing statistics are useful for load and traffic analysis, but
they are not causal evidence of expert roles. The next specialization test
requires saved final checkpoints and routing records.

For each matched model, compare:

1. normal routing;
2. leave-one-expert-out execution;
3. counterfactual replacement or rerouting;
4. matched random expert permutations;
5. cross-seed expert matching under the same intervention.

Evidence for functional specialization requires consistent expert-specific
effects on held-out task families, greater than a preregistered permutation
null, across seeds and task variants. Routing entropy, load CV, activation
frequency or byte-conditioned mutual information alone do not pass.

The minimum new artifact requirement is a checkpoint plus canonical routing
trace at the evaluation checkpoints. The existing expert-count package does
not currently provide those final states.

## Cognitive scaffold gates

The external scaffold should be tested before attempting to internalize it in
Crucible.

### Gate R1 — recursive computation

Compare one-pass generation with bounded:

`propose -> critique -> counterexample -> revise -> stop`

Use the same frozen model and answer/tool budget accounting. Primary outcomes
are held-out accuracy, correction success and cost-adjusted gain. Also report
contradiction rate, unsupported-claim rate and unnecessary-pass rate.

### Gate R2 — symbolic evidence

Factor symbolic retrieval and verification separately:

| Condition | Neural recursion | Symbolic evidence | Persistent write |
|---|---|---|---|
| A | no | no | no |
| B | yes | no | no |
| C | yes | yes | no |
| D | yes | yes | gated |

The primary question is whether structured evidence helps the system detect
and correct errors, not merely copy retrieved text.

### Gate R3 — typed persistence

CMS2 writes must distinguish at least verified fact, hypothesis, rejected
conclusion, unresolved question and procedure. Test interruption, restart,
contradiction, rollback and provenance failure. Compare memory-assisted
performance with post-update neural performance; they are different claims.

### Gate R4 — adaptive computation

A small controller may choose answer, retrieve, critique, falsify, create a
subgoal or stop. Compare it with fixed-depth, fixed-budget and deterministic
heuristic controllers. It passes only if decision overhead is included and
expected capability gain exceeds its cost.

### Gate R5 — goals

Evaluate goal generation and goal pursuit separately:

- goal quality, boundedness and relevance;
- progress per subgoal;
- execution success and completion rate;
- replanning after failure;
- recovery after interruption;
- false-progress and unsupported-completion rate.

Text that describes a plan is not evidence that a goal was maintained or
completed.

## Statistical and reporting rules

- Freeze manifests, splits, seeds, prompts and budgets before execution.
- Use at least two seeds for an initial gate and add a third seed before
  promoting a small effect as an architectural trend.
- Report per-seed results, not only means.
- Predeclare primary endpoints and tolerances.
- Preserve negative and failed results.
- Separate inference-time scaffold gains from learned-weight gains.
- Include cost, latency and tool use in any claim of practical improvement.
- Do not merge TinyStories language loss with cognitive-task scores into one
  unqualified intelligence number.

## Dependency-aware next steps

### Now: CPU-only preparation

1. Preserve the E4/E8/E16 metrics and initial anchors.
2. Mark causal specialization as blocked by missing final checkpoints and
   routing traces, rather than pretending the existing statistics answer it.
3. Finalize deterministic task generators, manifests and scoring contracts.
4. Audit the incomplete context-256/512 collection before using it for a
   context claim.

### Next paid or substantial GPU gate

Choose one, after review:

- a state-saving E4/E8/E16 replication for causal specialization; or
- the smallest matched-compute dense-versus-MoE quality comparison.

Do not start E32, recursive cognition, pretrained distillation and a new
context sweep simultaneously. Each would add a different confound.

### Later

Run the external recursive/symbolic scaffold on the sealed cognitive suite.
Introduce CMS2 persistence only after retrieval and verification are measured
separately. Internalize useful controllers only after the external process
shows a repeatable advantage.

## Decision standard

The project can claim an architectural improvement only when the result is:

1. replicated across seeds;
2. measured on held-out task structure;
3. compared with the simplest relevant control;
4. protected against data, capacity, tool and compute confounds; and
5. supported by mechanism-specific ablations.

Until then, use precise labels such as **language-modelling improvement**,
**sparse-capacity signal**, **execution optimization** or **scaffold-assisted
performance**. Reserve claims about reasoning, specialization, autonomy or
learning for gates that directly measure those properties.
