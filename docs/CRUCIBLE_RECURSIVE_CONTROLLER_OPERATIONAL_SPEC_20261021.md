# Crucible recursive reasoning controller: operational specification

Date: 2026-10-21
Status: design-only; no implementation, model training or cloud execution is authorized.

## Purpose and provenance

This document turns the conversation-provided recursive equations into a
bounded, inspectable controller specification. It compares them with the
symbolic substrate in `/home/timbushnell/Le_P2/Le_P3/cognitiion.MD` and the
measurement discipline in the BraidFormer trajectory-aware proposal.

The sources have different roles:

- the equations specify a proposed utility, value, memory and recursion policy;
- `cognitiion.MD` specifies typed symbolic state, contradiction handling,
  revision, simulation, planning and verification interfaces;
- BraidFormer specifies leakage controls, trajectory summaries, audit floors
  and compute accounting.

None of these sources demonstrates a learned metacognitive controller. This
document is an operational hypothesis and experiment plan.

## Equations preserved from the current exchange

The following equations are reproduced from the supplied recursive-reasoning
exchange. The implementation below must not silently replace them with a
different objective.

### Fixed-point reasoning objective

\[
\begin{aligned}
R^*=\operatorname{Fix}_R\Bigg\{\arg\max_r\Big[
&\mathbb E_{w\sim P(w\mid M,R)}
\big[U(r,w,G)+\gamma V(F(w,r,R))\big]\\
&+\lambda IG(r;M)\\
&+\rho C_{\text{consistency}}(r,R)\\
&+\sigma Q_{\text{self}}(r,R)\\
&-\beta H(P(w\mid M,R))\\
&-\kappa Cost(r)
\Big]\Bigg\}.
\end{aligned}
\]

### Recursive future value

\[
V(s)=\max_a\mathbb E_{s'\sim P(s'\mid s,a,R^*)}
\left[
U(a,s',G)+\gamma V(s')
+\lambda IG(a;s)-\kappa Cost(a)
\right]
\]

### Persistent memory update

\[
M_{t+1}=\operatorname{Compress}
\left(M_t\cup Evidence_t\cup R_t^*\right)
\]

### Recursive critique and evidence loop

\[
\Phi_0(R)=\operatorname{Reason}(R)
\]

\[
\begin{aligned}
\Phi_{k+1}(R)=\operatorname{Reason}\big(
&R,\\
&\operatorname{Critique}(\Phi_k(R)),\\
&\operatorname{Counterexample}(\Phi_k(R)),\\
&\operatorname{EvidenceSearch}(\Phi_k(R))
\big)
\end{aligned}
\]

\[
R^*=\operatorname{Fix}_R
\left\{\arg\max_k J(\Phi_k(R))\right\}
\]

with:

\[
\begin{aligned}
J(R)=&
\mathbb E[U+\gamma V]
+\lambda IG
+\rho C_{\text{consistency}}\\
&+\sigma Q_{\text{self}}
+\eta\Delta Competence_{\text{future}}\\
&-\beta H
-\kappa Cost.
\end{aligned}
\]

The proposed stopping conditions are convergence, low information gain, no
unresolved contradictions, or marginal reasoning benefit smaller than
marginal compute cost.

### Separate training objective proposal

\[
\mathcal L=
\mathcal L_{\text{answer}}
+\lambda_c\mathcal L_{\text{correction}}
+\lambda_v\mathcal L_{\text{verification}}
+\lambda_s\mathcal L_{\text{stopping}}.
\]

This is a possible future supervision objective, not part of the runtime
controller and not an authorization to train it.

## Canonical bounded controller state

The controller state must be explicit and serializable:

```text
ControllerState
  goal: Goal
  constraints: Constraint[]
  beliefs: BeliefState
  claims: Claim[]
  evidence_ledger: Evidence[]
  contradiction_graph: ContradictionGraph
  dependency_graph: DependencyGraph
  world_state: WorldState
  candidate_history: CandidateRecord[]
  self_metrics: SelfMetrics
  budget: ReasoningBudget
  memory_write_queue: MemoryProposal[]
  state_hash: digest
```

`BeliefState` is a finite, typed set of mutually exclusive or explicitly
related hypotheses with normalized probabilities only when calibration is
justified. A list of confidence scores is not automatically a probability
distribution.

`CandidateRecord` must retain the candidate reasoning result, actions taken,
observations, score components, verifier outcomes, cost and parent state hash.
This makes repeated self-confirmation and unverifiable memory writes
detectable.

`ReasoningBudget` includes maximum recursion depth, wall-clock time, action
count, evidence calls, counterexample calls, memory writes and total estimated
cost. Every child state inherits a reduced budget.

## Bounded state transition

At cycle (t), the controller performs:

1. Validate the current state hash, goal, constraints and budget.
2. Enumerate only allowed actions: answer, critique, counterexample search,
   evidence search, simulation, revision or stop.
3. Estimate each candidate's utility components using only information
   available before that candidate executes.
4. Select a candidate with a deterministic tie-break rule.
5. Execute it under a timeout and resource budget.
6. Update observations, contradiction edges, dependencies and self metrics.
7. Recompute exact checks and estimated scores separately.
8. Quarantine proposed memory writes until provenance and verification pass.
9. Stop or continue only if the stopping policy permits another cycle.
10. Serialize the resulting state and append an immutable transition record.

The controller may return `stop_unresolved` when the budget is exhausted or
contradictions remain. It must not convert an unresolved state into a verified
fact merely to satisfy the goal.

## Operational definitions and measurement contracts

| Term | Required inputs | Observable output | Candidate estimator | Exactness and calibration |
| --- | --- | --- | --- | --- |
| (R) | Claims, goals, constraints, world state, evidence | Canonical state digest | Typed serialized state | Exact structure and hash |
| (r) | Current state and allowed action set | Candidate answer/plan/revision | Enumerated controller action | Exact membership in allowed set |
| (w) | Action and environment/verifier result | Observation or successor state | Task oracle, simulator or tool result | Exact only with a trusted oracle |
| (U) | Goal, constraints, outcome and risk | Scalar utility components | Pre-registered task score plus risk penalties | Must be calibrated against task success; never free-form self-score |
| (V) | State, action, successor distribution | Expected downstream value | Bounded branch simulation or learned predictor | Estimate; require held-out calibration |
| (IG) | Belief distribution before/after evidence | Uncertainty reduction | (H(B_t)-\mathbb E[H(B_{t+1})\mid a]) | Exact only for a calibrated finite belief model |
| (C_{\text{consistency}}) | Contradiction graph and constraints | Violation/consistency score | Explicit constraint checks and weighted conflicts | Graph violations can be exact; normalization is a design choice |
| (Q_{\text{self}}) | Historical predictions and outcomes | Reliability estimate | Brier score, calibration error or held-out success rate | Must use prior outcomes, not current self-description |
| (H) | Calibrated belief distribution | Entropy | (-\sum_i p_i\log p_i) | Exact calculation only after calibration |
| `Cost` | Measured time, tool calls, memory and risk | Resource cost | Instrumented wall time/CUDA time plus declared penalties | Measured components exact; future cost predictive |
| (\Delta Competence_{\text{future}}) | Learning intervention and later-task set | Change on held-out future tasks | Post-intervention score minus no-intervention control | Requires separate future-task protocol and matched cost |
| `Compress` | Evidence, result, provenance, contradictions | Memory proposal or committed record | Typed consolidation and deduplication | Must preserve provenance and rollback links |
| `Fix_R` | Candidate sequence and state hashes | Stable candidate/state or bounded failure | Convergence detector plus cycle/oscillation checks | Operational approximation, not mathematical fixed-point proof |

### Information gain

Use a finite belief state for the first implementation. For hypotheses
(B_t=\{(b_i,p_i)\}), define:

\[
H(B_t)=-\sum_i p_i\log p_i
\]

For action (a), estimate the posterior distribution from a trusted evidence
model or verifier. If no calibrated likelihood model exists, `IG` must be
reported as unavailable rather than fabricated from a confidence change.

### Future competence

Define a held-out future-task distribution (D_{future}) that is not used for
controller selection. After an explicitly declared learning intervention,
measure:

\[
\Delta Competence_{future}
= Score(D_{future};\text{after})
- Score(D_{future};\text{no-intervention control})
\]

Report token, optimizer-update, wall-time and tool costs. Current-task answer
quality is not evidence of future competence.

## Failure modes and safeguards

### Oscillation and repeated self-confirmation

- reject a transition if the same state/action pair repeats without new
  evidence;
- detect cycles using recent state hashes;
- require a new evidence identifier before retrying an unchanged critique;
- cap repeated revisions of the same claim;
- retain rejected candidates for audit rather than silently deleting them.

### Unverifiable memory writes

- separate `memory_proposal` from `memory_commit`;
- require source, timestamp, evidence hash, confidence, contradiction links and
  verifier result;
- write hypotheses as hypotheses, not facts;
- reject writes with missing provenance or unresolved contradictions;
- support transactional rollback and append-only audit records.

### Runaway recursion

- maximum depth and wall-clock budget;
- maximum evidence, counterexample and tool calls;
- maximum state size and memory-write rate;
- stop on non-finite scores or invalid state transitions;
- return an explicit unresolved status on budget exhaustion.

### Miscalibrated confidence and information gain

- calibrate on held-out tasks;
- report Brier score, expected calibration error and selective-risk curves;
- compare against a fixed-confidence baseline;
- never use a self-reported confidence value as independent evidence;
- retain a random audit fraction of actions classified as low utility.

## Baselines and controlled comparison

The controller must beat simple policies under the same task, tool and answer
budgets to justify its complexity:

1. **One-pass:** answer once and stop.
2. **Fixed-depth:** always perform 2 or 4 bounded reasoning passes.
3. **Confidence stop:** continue only while calibrated confidence is below a
   preregistered threshold.
4. **Verification-only:** answer, run the verifier, revise once if rejected.
5. **Adaptive controller:** use the operationalized utility, information gain,
   consistency, self-quality and cost terms.

Primary metrics:

- exact task accuracy and held-out composition/depth accuracy;
- correction success and false-correction rate;
- contradiction resolution and unresolved-state rate;
- memory precision, rollback success and provenance validity;
- reasoning actions, wall time, tool calls and peak memory;
- controller overhead and stop calibration.

The adaptive controller passes only if it improves accuracy at matched total
cost, or preserves accuracy while reducing total cost, across preregistered
seeds and task families. A gain over one-pass alone is insufficient if a
fixed-depth or verification-only policy performs as well.

## C0/C1 operational addendum (2026-10-21)

This addendum is preregistered for the CPU-only C0 contract simulator and C1
frozen-task replay. It resolves four risks before any model evaluation.

### Calibrated uncertainty and entropy safeguards

The entropy term is not a license to reward confident predictions. A confidence
value is treated as a probability only when it comes from a declared finite
belief model whose calibration was established on data independent of policy
selection. C0/C1 must report calibration status, Brier score and selective
risk separately. If a candidate has no calibrated posterior and an action has
no trusted likelihood model, its information gain and entropy reduction are
`unavailable`, not zero-quality evidence silently converted into utility.

An entropy decrease receives positive credit only when it is supported by a
verified outcome or a preregistered calibrated evidence transition. A
confident-but-wrong candidate therefore cannot improve its score merely by
claiming certainty. Independent verification remains able to reject such a
candidate, and a contradiction blocks a factual memory commit.

### Units, normalization and coefficients

Every utility component is normalized to a declared dimensionless range before
coefficient application. C0/C1 use the fixed coefficient vector recorded in
the result manifest; coefficients are not tuned after seeing policy outcomes.
Measured resource cost is recorded in native units (action count, tool calls,
evidence calls, estimated cost units and wall time) and mapped to a declared
bounded penalty. The manifest records the coefficient vector, normalization
rules, seed and policy configuration so a later run can reproduce the exact
comparison.

The operational C0/C1 vector is:

\[
(w_U,w_{IG},w_C,w_Q,w_H,w_{Cost},w_{Future})
=(1.0,0.25,0.50,0.25,0.10,0.10,0.0).
\]

The future-competence coefficient is explicitly zero for C0/C1. It must not
be replaced by a current-task heuristic and must remain disabled unless an
independent future-task intervention protocol is available.

### Contract validation versus policy advantage

C0/C1 have two separate conclusions. Contract success means that typed state,
budget inheritance, transition hashing, cycle detection, independent
verification and transactional memory rules behave as specified. Policy
advantage means that a policy improves outcomes on imperfect candidates and
misleading/contradictory evidence at matched action, tool and cost budgets.
Passing the first does not imply the second. Synthetic replay may establish
implementation correctness and expose failure modes, but it is not evidence
that a neural or adaptive policy is cognitively superior.

### C0/C1 execution boundary

C0/C1 use only deterministic CPU code, frozen v2 task manifests and synthetic
candidate/evidence responses. They do not train or download a model, alter
Crucible's neural architecture, access a GPU or spend cloud credit. The
future-competence term is reported as disabled, and failed policy hypotheses
are preserved rather than tuned away.

## Falsifiable hypotheses

- **H1 — Adaptive effort:** a bounded adaptive policy achieves better
  held-out accuracy than fixed depth at equal total compute.
- **H2 — Information value:** calibrated information gain improves action
  selection beyond confidence-only stopping.
- **H3 — Verification:** independent verification reduces false corrections
  and false memory commits relative to self-critique alone.
- **H4 — Future competence:** evidence and revision interventions improve a
  later held-out task distribution relative to a no-intervention control.
- **H5 — Safety:** provenance gates and cycle limits prevent invalid memory
  commits and runaway recursion without unacceptable task-completion loss.

Failure of H1 or H2 means the simpler policy should be retained. Failure of
H4 means current-task improvement must not be described as competence growth.

## CPU-only preparation sequence

### C0 — Contract simulator

Create synthetic typed claims, contradictions, evidence records and finite
belief states. Verify state hashes, weighted revision, dependency rollback,
entropy/IG calculations, cycle detection, budget exhaustion and transactional
memory writes. This tests the controller contracts without a neural model.

### C1 — Frozen-task replay

Replay corrected v2 cognitive manifests with exact oracles and compare the
five baseline policies. Use deterministic synthetic candidate outputs first;
do not train or download a model. The purpose is to expose scoring, stopping
and leakage errors before model evaluation.

### C2 — Model evaluation only

After C0/C1 pass and separate approval is granted, evaluate a fixed competent
model on the same tasks. Keep weights frozen while comparing one-pass,
fixed-depth, confidence, verification-only and adaptive policies.

### C3 — Learning intervention

Only after future-competence measurement is specified should the separate
training objective be considered. It requires an untouched future-task set,
no test-set checkpoint selection and full cost accounting.

## Recommendation

The older cognition architecture should be reused as an explicit symbolic
substrate, not merged wholesale into Crucible's neural core. The equations
should first be tested as an external controller over that substrate. The
BraidFormer trajectory controller remains a separate adaptive-training
baseline until its own target and benefit are established.

No model training, checkpoint download, neural architecture change or cloud
spending is authorized by this specification.
