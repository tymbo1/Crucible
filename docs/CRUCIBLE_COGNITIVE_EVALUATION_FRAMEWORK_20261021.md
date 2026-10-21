# Crucible cognitive evaluation framework

Date: 2026-10-21  
Status: design-only; no benchmark training, checkpoint download, cloud rental or frozen-artifact modification authorized.

## Purpose

Crucible currently has language-modelling evidence and preliminary sparse-capacity evidence. Those measurements are necessary, but they do not establish mathematical reasoning, abstraction, metacognition, planning, autonomy or persistent learning.

This framework keeps language competence as one metric while adding task families that test the capabilities the project is intended to investigate. It also separates capability from architectural causality: a better score is not automatically evidence that MoE routing, recursion, symbolic memory or a pretrained initialization caused the improvement.

## Current evidence boundary

The E4/E8/E16 matched-active-compute results support a narrow claim:

> On the tested byte-level TinyStories-style task, increasing sparse capacity improved validation NLL/BPB in the recorded two-seed experiment.

They do not establish:

- functional or transferable expert specialization;
- mathematical, logical or relational reasoning;
- metacognitive error detection;
- autonomous goal generation or goal pursuit;
- long-context generalization;
- superiority at equal total parameters, wall time, memory or cost;
- that the grouped execution backend improves model capability.

The CPU-only checkpoint inventory found initial seed anchors and metrics for the expert-count runs, but no trained final E4/E8/E16 checkpoints or routing traces suitable for causal expert-ablation studies. The existing specialization analysis measures byte-conditioned routing-mass differentiation and explicitly does not claim semantic or causal specialization. A causal specialization gate therefore requires a future run that persists final checkpoints, routing traces and intervention metadata.

## Capability dimensions

| Dimension | Primary question | Required evidence | Invalid shortcut |
|---|---|---|---|
| Language competence | Does the model predict and generalize language? | EOS-excluded byte NLL/BPB, held-out stories, exposure and compute accounting | Treating lower TinyStories loss as general intelligence |
| Mathematics | Can it apply rules to unseen instances? | Exact answers, proof/checker validity, compositional and extrapolation splits | Memorized templates or answer-string likelihood alone |
| Logic | Can it maintain consistency and derive consequences? | Formal validity, contradiction detection, proof completion and counterexample tests | Self-described confidence without correctness correlation |
| Relational cognition | Can it bind, retrieve and compose relations under distraction? | Delayed key-value retrieval, entity tracking, multi-hop composition, interference and length extrapolation | Routing entropy or attention maps without task intervention |
| Metacognition | Does uncertainty predict errors and trigger useful correction? | Calibration, selective accuracy, error-detection AUROC, correction success and abstention utility | Saying “I may be wrong” |
| Creativity | Can it produce novel, coherent, useful constrained solutions? | Blind constraint scoring, novelty against training data, usefulness and independent human/evaluator agreement | Subjective fluency or novelty without utility |
| Planning | Can it execute and revise multi-step plans? | Dependency validity, task success, cost, replanning after failure and partial-progress measures | Textual plans that are never executed |
| Autonomy | Can it preserve a bounded objective across time and interruption? | Goal persistence, subgoal quality, progress, completion, restart recovery and false-progress rate | Generating a plausible goal once |
| Continual learning | Can new validated information persist without damaging old competence? | Retention, forward transfer, backward interference, rollback and provenance checks | Memory retrieval counted as weight learning |

## Task design rules

Each task family must have a generator or immutable manifest, separate train/validation/test instances and a known evaluation procedure. Where possible, the test split must contain new combinations of familiar primitives rather than merely new surface strings.

### Minimum task families

1. **Language:** retain the existing byte-level TinyStories evaluation for continuity; add held-out story structure and longer-context splits when the context gate is complete.
2. **Mathematics:** arithmetic and algebra with held-out values, compositions and rule combinations; use exact symbolic checkers where available.
3. **Logic:** deduction, contradiction detection, entailment and proof-step completion with generated certificates.
4. **Relational memory:** delayed retrieval, distractor insertion, relation reversal, multi-hop composition and length extrapolation.
5. **Planning:** small executable environments with dependencies, resource limits, injected failures and replanning opportunities.
6. **Metacognition:** answerable and unanswerable items with calibrated confidence, forced abstention and a second-pass correction budget.
7. **Autonomy:** interrupted bounded objectives requiring state restoration, progress accounting and explicit completion criteria.

The first version should be small enough to run on CPU for validator development. It should not be presented as a capability result until a model has received sufficient task-specific curriculum.

## Training sufficiency and evaluation modes

Failure on an unfamiliar task can reflect missing curriculum, insufficient optimization, inadequate capacity or an architectural limitation. Every cognitive result must identify which of these explanations the experiment can exclude.

Use three distinct modes:

### Mode A — frozen-model probes

Use existing checkpoints to test whether a capability is already present. Report this as probing only. Do not interpret a failure as an architectural impossibility.

### Mode B — controlled task training

Train all compared systems on the same generated curriculum, raw-byte/token exposure, optimizer-update budget and task mixture. Evaluate on held-out instances, held-out combinations and, where possible, length or rule extrapolation.

### Mode C — scaffold ablations

Add recursion, retrieval, symbolic verification, CMS2 or adaptive compute one factor at a time. Compare against the same neural model with matched answer, tool, token and wall-time budgets.

Pretrained initialization is a separate enabling ablation. It may reduce the language-competence burden, but it must not replace the from-scratch control or be used to attribute teacher knowledge to a new Crucible mechanism.

## Causal comparison requirements

For every architectural claim, record and match or explicitly account for:

- total and active parameters;
- expert count, expert width and routing policy;
- training bytes/tokens and optimizer updates;
- tokenizer and vocabulary;
- context length and batch construction;
- initialization and pretrained status;
- optimizer, precision and deterministic settings;
- neural, tool and retrieval calls;
- wall time, GPU time, peak VRAM and monetary cost;
- task curriculum and information available at evaluation;
- checkpoint, RNG, sample-order and provenance state.

The core comparisons should include, where feasible:

1. a simple dense control at matched active compute;
2. E4, E8 and E16 sparse variants under the same task curriculum;
3. a fixed-total-parameter control where the question requires it;
4. recursion-only, retrieval-only and full hybrid scaffolds;
5. a no-memory or stateless restart control for persistence claims.

## Metacognition and autonomy acceptance rules

Metacognition passes only when confidence predicts correctness better than a declared baseline and an additional critique/revision step improves outcomes at a measured cost. Self-reported uncertainty is not sufficient.

Autonomy passes only when the system maintains a bounded objective across multiple steps, selects useful subgoals, produces measurable progress, recovers after interruption and distinguishes completion from failure. Goal generation quality and goal-pursuit success must be scored separately.

Persistent memory passes only when writes are typed and provenance-bearing, rejected or unresolved states remain distinguishable from verified facts, rollback works, and later performance improvements are separated into retrieval benefit versus neural-weight change.

## Research sequence

### Gate A — Evaluation infrastructure, CPU-only

- Freeze task schemas, generators, manifests and scoring code.
- Add deterministic validators, exact checkers and provenance records.
- Build a small baseline report without downloading models or running training.
- Preserve the existing NLL/BPB reports as language-only baselines.

### Gate B — Neural architectural baselines

- Finish the context-collection audit before drawing long-context conclusions.
- Preserve E4/E8/E16 metrics and add final-state persistence in the next authorized training package.
- Run causal expert interventions only after trained checkpoints and routing traces exist.
- Compare sparse variants with a dense matched-compute control on task families they have actually been trained to solve.

### Gate C — Relational and adaptive computation

- Test long-distance binding and compositional generalization independently of recursive scaffolding.
- Evaluate bounded recursive critique against additional neural recursion alone.
- Add symbolic retrieval and verification as separate ablations.

### Gate D — Persistence and goals

- Add CMS2 only after retrieval/verification benefits are measured separately.
- Test interruption, restart, rollback, false progress and provenance.
- Add goal generation before claiming goal pursuit; measure both independently.

### Gate E — Internalization

Only after an external scaffold shows repeatable held-out gains should useful controllers or memory policies be trained into Crucible. Compare frozen scaffold, trained controller and from-scratch controls.

## First decisive cognitive gate

The first narrow cognitive experiment should be:

> Does frozen neural generation plus bounded symbolic critique/retrieval improve held-out, verifiable problem solving over additional neural recursion alone at equal or measured compute cost?

This is downstream of CPU-only benchmark design and task-curriculum preparation. It should not be combined with expert-count changes, pretrained initialization, CMS2 persistence and autonomous goals in one run.

## Reporting standard

Every result must state:

- what was measured;
- what was held constant;
- what information the system had;
- whether the task was trained, probed or held out;
- the simpler baseline it beat;
- compute and tool costs;
- failure and contradiction rates;
- which claims remain hypotheses.

The project should use “language-model improvement,” “sparse-capacity result,”
“execution optimization,” “cognitive capability result” and “autonomous
behaviour result” as separate labels. A result may satisfy one label without
satisfying any of the others.

## Immediate decision

Do not start a new Crucible training run or rent cloud hardware from this
framework alone. Finish the CPU-only evaluation registry and evidence audit,
then return a preregistered smallest-decisive gate for approval.
