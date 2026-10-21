# Crucible cognition architecture comparison

Date: 2026-10-21
Status: CPU-only source audit and experiment proposal; no execution authorized.

## Executive result

Three verifiable design threads are related but not interchangeable:

1. `Le_P2/Le_P3/cognitiion.MD` describes a symbolic, stateful runtime for
   perception, inference, contradiction handling, planning, action and
   revision.
2. BraidFormer describes a trajectory-aware training-policy controller that
   predicts the learning utility of a candidate batch and gates computation.
3. Crucible's current roadmap describes controlled neural evaluation of
   recursive critique, symbolic evidence, persistent state, adaptive compute
   and goal pursuit.

The older symbolic runtime is the closest existing implementation design for
the operational cognition loop. The BraidFormer proposal supplies useful
resource-allocation and audit principles, but it is not a reasoning engine.
Crucible currently contains evaluation and instrumentation plans, not a
demonstrated integrated cognition runtime.

The recursive fixed-point, future-value and training-loss equations quoted in
the current conversation remain proposed research formulations unless their
original exchange is independently recovered. They must not be attributed to
the files below.

## Provenance audit

| Source | Verified location | Evidence status |
| --- | --- | --- |
| Symbolic cognition design | `/home/timbushnell/Le_P2/Le_P3/cognitiion.MD` | Present; dated 2026-02-12; conceptual design with Python skeletons and interfaces |
| BraidFormer trajectory policy | `/home/timbushnell/braidformer_lab/docs/TRAJECTORY_AWARE_TRAINING_PROPOSAL.md` | Present; explicitly deferred; no implementation or training authorized |
| Crucible roadmap | `docs/CRUCIBLE_RESEARCH_ROADMAP_20261021.md` | Present; research plan, not capability evidence |
| Crucible evaluation framework | `docs/CRUCIBLE_COGNITIVE_EVALUATION_FRAMEWORK_20261021.md` | Present; benchmark and gate design, not capability evidence |
| Claimed `braid_exchange_2026-10-14_0721.txt` | Not found under `/home/timbushnell` | Unverified; do not cite as recovered source |
| Claimed `braid_exchange_2026-10-14_0844_recursive_synthesis.txt` | Not found under `/home/timbushnell` | Unverified; do not cite as recovered source |
| Actual `braid_exchange_2026-02-26_1434.txt` | `/home/timbushnell/Documents/Projectbrain/` | Present, but concerns ontology ingestion and `is_a` relations, not the quoted equations |
| Actual `braid_memory_exchange.txt` | `/home/timbushnell/Documents/Projectbrain/` | Present, but only a short memory-related excerpt |

The claimed October filenames may exist outside the inspected workspace, but
there is currently no local evidence for them. The quoted equations are
therefore recorded here as conversation-origin proposals, not recovered file
contents.

## Mechanism comparison

### 1. Older symbolic cognition runtime

The design defines four layers:

- perceptual grounding, producing anchors, claims and confidence tags;
- practical action, with goals, constraints and checked effects;
- abstract reasoning, with claims, inference and a contradiction graph;
- temporal planning, with time-indexed world states, plans and
  counterfactual branches.

Its core cycle is:

```text
Perceive -> Update -> Reason -> Simulate -> Act -> Verify -> Revise -> Repeat
```

The file defines or sketches the following state and interfaces:

- `Claim`, `Goal`, `Constraint`, `WorldState`, `Plan`, `PlanStep` and
  `Counterfactual` records;
- a `ContradictionGraph` and weighted revision operator;
- a dependency graph for cascading retractions;
- symbolic memory with an active/retracted view;
- a `SelfModel` containing basic cycle statistics;
- Prolog and Lisp bridge contracts.

The strongest concrete mechanism is minimal-change revision: when claims
conflict, retract weaker commitments and preserve stronger or invariant
commitments. The file also includes branching simulation and verification
hooks.

What it does not establish is a trained neural implementation, a validated
goal generator, a learned stopping policy, or autonomous long-horizon task
completion. Most of the document is a design and interface specification.

### 2. BraidFormer trajectory-aware training

BraidFormer addresses a different control problem. Its compact trajectory
state (z_t) contains detached summaries such as recent loss, gradient
statistics, validation loss, optimizer progress, exposure and forgetting.
Given a candidate batch (x_{t+1}), it proposes:

\[
\widehat{u}_{t+1}=P_\phi(z_t,x_{t+1})
\]

where the target is expected reducible loss or learning utility. The policy
may fully update, downweight or audit a batch, but must retain coverage and an
exploration floor.

Its useful mechanisms are:

- pre-update prediction rather than post-update leakage;
- token-matched and compute-matched comparisons;
- false-skip measurement;
- fixed audit fractions and coverage floors;
- predictor checkpointing and trajectory-state hashes;
- explicit accounting for controller overhead.

It does not provide symbolic beliefs, contradiction revision, evidence
provenance, world-state simulation or persistent user goals. It should remain
a separate adaptive-compute workstream until its own baseline is established.

### 3. Crucible roadmap and evaluation framework

Crucible's current design separates the proposed mechanisms into gates:

- recursive passes: propose, critique, counterexample, revise and stop;
- symbolic evidence and verification;
- typed persistent state/CMS2;
- adaptive computation;
- goal generation and goal pursuit;
- continual learning and retention.

The framework correctly separates goal quality from goal completion, memory
assistance from neural learning, and language loss from cognitive capability.
The corrected v2 task manifests provide arithmetic, logic, relational and
error-correction probes with composition/depth holdouts. The instrumentation
pilot adds checkpoint, optimizer, RNG, routing-probe and intervention
requirements, but has not been run.

## Compatibility matrix

| Mechanism | Symbolic runtime | BraidFormer policy | Crucible integration | Main risk |
| --- | --- | --- | --- | --- |
| Typed claims and provenance | Native design | Not present | CMS2/evidence factor | Unverified writes can look like learning |
| Contradiction detection/revision | Native design | Not present | Recursive verification gate | Revision can become destructive without dependency tracking |
| Counterexample search | Proposed hook | Not present | Recursive task factor | Textual counterexamples may be plausible but invalid |
| World-state simulation | Native design | Not present | Planning/goal gate | Simulation accuracy and branching cost |
| Persistent state | Symbolic memory and active view | Trajectory state | State-saving pilot/CMS2 | Confusing retrieval with changed neural competence |
| Self-evaluation | Basic cycle statistics | Predictor calibration and false skips | Metacognition gate | Stated confidence may not predict error |
| Adaptive compute | Not fully specified | Batch gating | Answer/retrieve/critique/stop controller | Controller overhead and biased coverage |
| Goal generation/pursuit | Goal data contract | Not present | Separate Crucible gates | Goal prose is not goal completion |

## Conversation-origin equations

The equations below are useful hypotheses, but their original file provenance
has not been recovered in this workspace:

\[
R^*=\operatorname{Fix}_R\left\{\arg\max_r\left[
\mathbb{E}[U+\gamma V]+\lambda IG+\rho C_{\mathrm{consistency}}
+\sigma Q_{\mathrm{self}}-\beta H-\kappa Cost(r)\right]\right\}
\]

\[
V(s)=\max_a\mathbb{E}_{s'}[U(a,s',G)+\gamma V(s')+\lambda IG(a;s)-\kappa Cost(a)]
\]

\[
M_{t+1}=\operatorname{Compress}(M_t\cup Evidence_t\cup R_t^*)
\]

and the recursive revision proposal:

\[
\Phi_{k+1}(R)=\operatorname{Reason}\left(R,
\operatorname{Critique}(\Phi_k),
\operatorname{Counterexample}(\Phi_k),
\operatorname{EvidenceSearch}(\Phi_k)\right).
\]

Before implementation, every term needs an operational definition, a bounded
compute budget, a source of ground-truth feedback and a measurable stopping
rule. In particular, `future competence` cannot be claimed from a current
answer score; it requires held-out later tasks or a pre-registered transfer
test.

## Smallest falsifiable experiment sequence

### Gate C0 — CPU-only runtime contract

Use synthetic typed claims and the corrected v2 task records to test:

1. claim insertion, contradiction detection and weighted revision;
2. dependency-cascade retraction;
3. provenance and confidence preservation;
4. world-state/plan serialization and reload;
5. deterministic stop conditions and state hashes.

This tests the symbolic machinery without training a model. It should produce
no claim about neural reasoning.

### Gate C1 — Instrumentation pilot

Use the already prepared [cognitive instrumentation pilot](CRUCIBLE_COGNITIVE_INSTRUMENTATION_PILOT_20261021.md).
First verify learnability, checkpoint/RNG/probe capture and intervention
replay. Do not compare recursive mechanisms until the state and task pipeline
passes its mechanical gate.

### Gate C2 — External recursive scaffold

Keep Crucible weights fixed and compare, at matched answer and tool budgets:

1. one-pass answer;
2. answer plus critique/revision;
3. answer plus independently checked counterexample/evidence;
4. fixed-depth versus bounded stop policy.

Use exact task oracles where available. Report accuracy, correction rate,
false corrections, contradiction resolution, compute cost and stop decisions.
This isolates the process before attempting to internalize it in the neural
architecture.

### Gate C3 — Persistent-state interruption test

Compare stateless restart, serialized symbolic state and serialized neural
state on the same unfinished objective. Measure resumed-task completion,
provenance validity, rollback correctness and storage cost. Do not call a
CMS2 retrieval gain neural learning.

### Gate C4 — Adaptive computation

Only after C2 has a measurable benefit, train or fit a small controller to
choose answer, retrieve, critique, falsify, subgoal or stop. Compare against
fixed-depth and fixed-budget policies, including controller overhead and an
audit floor.

The BraidFormer trajectory predictor should remain a separate arm until this
controller has a clear target. Combining both controllers first would make
credit assignment and failure diagnosis ambiguous.

## Recommendation

The smallest next action is CPU-only Gate C0 plus a source/provenance record.
Do not merge `cognitiion.MD` wholesale into Crucible. Reuse its typed state,
contradiction, dependency, revision and verification concepts selectively.
Reuse BraidFormer's leakage, audit-floor, calibration and compute-accounting
discipline. Use Crucible's corrected v2 manifests and evaluation framework as
the common measurement layer.

No model training, checkpoint download, architecture modification or cloud
spending is authorized by this comparison.
