# Crucible research roadmap

Date: 2026-10-21  
Status: design-only; proposed mechanisms are hypotheses, not demonstrated capabilities.

## Research thesis

Crucible is the neural substrate. The long-term research question is whether a
small sparse neural system can become part of a larger learning architecture
that maintains unresolved objectives, allocates computation, uses verifiable
evidence, preserves useful state across interruptions and improves its future
problem-solving process.

The system must be evaluated as a set of controlled components rather than as
one undifferentiated claim of intelligence.

The multidimensional evaluation protocol is defined in
[`CRUCIBLE_COGNITIVE_EVALUATION_FRAMEWORK_20261021.md`](CRUCIBLE_COGNITIVE_EVALUATION_FRAMEWORK_20261021.md).
NLL/BPB remains the language-competence measure; it is not the sole measure of
architectural progress.

## Proposed components

1. **Crucible neural substrate** — learned representations, sparse experts,
   attention and memory mechanisms.
2. **Recursive cognition** — hypothesis, critique, counterexample,
   verification and revision.
3. **Symbolic reasoning** — relations, logic, evidence and provenance.
4. **CMS2 consolidation** — typed persistent state for verified facts,
   hypotheses, rejected conclusions, procedures and unresolved questions.
5. **Autonomous goal engine** — bounded goal generation, planning, execution,
   monitoring and review.
6. **Continual learning** — controlled updates that improve competence without
   catastrophic forgetting or contamination of established knowledge.
7. **Meta-learning** — experience improving the strategy used on later,
   unfamiliar tasks.

These components may interact eventually, but each must first beat a simpler
baseline under a fair compute and information budget.

## Core distinctions to preserve

### Goal generation versus goal pursuit

A system may generate plausible objectives without making progress on them.
Measure the quality and boundedness of generated subgoals separately from the
ability to execute, monitor, revise and complete them.

### Persistent memory versus neural learning

CMS2 retaining a verified result is not the same as Crucible changing its
weights or improving its underlying competence. Report memory-assisted
performance and post-update neural performance as separate outcomes.

### Retrieval versus correction

The important hybrid test is not whether the model copies a retrieved answer.
It is whether structured evidence helps it detect, explain and correct an
otherwise wrong prediction.

### Described reasoning versus executed reasoning

Text that describes a plan is not evidence that the system maintained a goal,
selected a useful experiment, evaluated the result or resumed after an
interruption. Those behaviours require instrumented task execution.

## Research tracks and falsifiable questions

| Track | Question | Minimum baseline |
|---|---|---|
| Recursive reasoning | Do additional bounded reasoning passes improve unseen-task accuracy? | One-pass Crucible at matched answer/tool cost |
| Learned verification | Can the system identify and correct errors without being handed the answer? | Unverified self-critique and external verifier controls |
| Persistent cognitive state | Does typed state survive interruption and improve later task completion? | Stateless restart and untyped-memory controls |
| Neuro-symbolic integration | Do symbolic constraints improve generalization rather than only retrieval? | Neural-only and retrieval-only controls |
| Adaptive compute | Can a controller select answer, retrieve, critique, falsify, subgoal or stop actions profitably? | Fixed-depth and fixed-budget policies |
| Autonomous goal generation | Can unresolved objectives produce useful, bounded subgoals? | Human-written subgoals and unconstrained generation controls |
| Goal pursuit | Can the system execute and revise a subgoal to completion? | Open-loop plan execution |
| Continual learning | Can new validated information persist without damaging prior competence? | Isolated retraining and replay controls |
| Meta-learning | Does experience improve later unfamiliar-task strategy? | Same-task repetition and no-history controls |

The measurement design for these tracks is specified in
[`CRUCIBLE_COGNITIVE_EVALUATION_FRAMEWORK_20261021.md`](CRUCIBLE_COGNITIVE_EVALUATION_FRAMEWORK_20261021.md).
That framework is a design specification, not evidence that any listed
capability has been demonstrated.

## Recommended sequencing

### Phase 0 — Baseline and evidence discipline

- Preserve the from-scratch E16 Branch B baseline.
- Keep the grouped execution backend opt-in until its exact equivalence scope
  is documented.
- Freeze task sets, compute budgets, provenance and checkpoint rules.
- Keep architectural claims separate from execution-speed claims.

### Phase 1 — External recursive scaffold

Use frozen Crucible weights and a bounded controller implementing:

`propose -> critique -> counterexample -> evidence review -> revise -> stop`

Start on verifiable tasks with held-out problems. Measure accuracy, correction
success, contradiction rate, unsupported claims, number of passes and total
neural/tool tokens. This tests the process before changing the architecture.

### Phase 2 — Symbolic verification and typed CMS2

Add symbolic retrieval and verification as independent factors. CMS2 writes
must carry epistemic type, provenance, confidence, contradiction links and
rollback information. Compare baseline, recursion-only, hybrid retrieval and
full gated persistence conditions.

### Phase 3 — Learned compute allocation

Train a small controller to choose among prevalidated actions. It must be
compared with fixed-depth, fixed-budget and deterministic heuristic policies.
The controller passes only if its net accuracy or correction benefit exceeds
its added compute and decision latency.

### Phase 4 — Goal generation and pursuit

Introduce unresolved objectives and bounded subgoal generation. Evaluate goal
quality, progress per subgoal, completion rate, recovery after interruption,
and false-progress rate separately.

### Phase 5 — Controlled neural internalization

Only after the external scaffold demonstrates repeatable gains should useful
parts be trained into Crucible. Compare frozen-scaffold, trained-controller
and from-scratch controls so persistence or external tools are not mistaken
for learned capability.

## Pretrained foundation as an enabling ablation

Pretrained initialization is an optional way to reduce the language-competence
burden; it is not a replacement for the architectural tracks above. The
current feasibility plan recommends a frozen SmolLM2-135M teacher for a small
distillation pilot, with the existing from-scratch model retained as the
primary control. FFN transplantation remains a separate, higher-confound
route.

Any pretrained pilot must report teacher cost, tokenizer conversion, held-out
generalization and whether the gain survives when measured at matched raw-byte
exposure. See the comparative initialization plan for the preregistered pilot
proposal.

## Acceptance principles

Every mechanism must satisfy all applicable gates:

- improvement over a simpler baseline on held-out tasks;
- matched or explicitly accounted compute, information and tool budgets;
- reproducible seeds and immutable task manifests;
- provenance-correct evidence and typed memory writes;
- no increase in unsupported claims or unresolved contradictions hidden by
  aggregate accuracy;
- interruption, restart and rollback tests where persistent state is involved;
- separate accounting for memory-assisted performance and learned-weight
  improvement.

## Compute and cloud policy

Advertised GPU prices are leads, not measured availability or authorization.
No cloud run should begin until tenancy/privacy, environment reproducibility,
artifact transfer, maximum spend, stop conditions and result collection are
specified. Local CPU-only design and synthetic validation remain preferred
until a paid run is justified by a preregistered gate.

## Current next gate

Do not combine all proposed mechanisms in one experiment. The first narrow
architectural question should be:

> Does frozen Crucible plus bounded recursive critique and verification improve
> held-out, verifiable problem solving over additional neural recursion alone
> at equal or measured compute cost?

If that succeeds, add symbolic retrieval. Add CMS2 persistence only after the
retrieval/verification contribution is separately measured. Delay neural
internalization and autonomous goal pursuit until the external process has
demonstrated a repeatable advantage.

Before any substantial cognitive run, complete the CPU-only evidence audit and
prepare the task manifests and scoring contracts in the cognitive evaluation
framework. The existing E4/E8/E16 package has metrics and initial anchors but
no trained final checkpoints or routing traces for causal specialization
interventions.

The current preparation artifacts are:

- `docs/CRUCIBLE_CONTEXT_GATE_AUDIT_20261021.md` — complete/partial context
  coverage and verified metric hashes;
- `docs/CRUCIBLE_STATE_SAVING_REPLICATION_PLAN_20261021.md` — staged sparse
  replication, dense-control decision, checkpoint schema and cost envelope;
- `tools/cognitive_task_suite.py` and `tools/build_cognitive_manifests.py` —
  deterministic CPU task generators, exact oracles and immutable manifests.
