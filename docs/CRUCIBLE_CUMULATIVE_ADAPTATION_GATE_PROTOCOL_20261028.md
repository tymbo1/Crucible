# Crucible verified-experience-to-acquisition gate v1

Date: 2026-10-28
Status: **UNTESTED — protocol proposal only; no implementation, training or execution approved**

## Objective

Test whether verified experience changes the learner's future acquisition
strategy, rather than merely adding procedures to a retrievable catalogue.

The proposed first gate is deliberately model-agnostic. It tests the causal
mechanism of cumulative adaptation in a constrained proposal/search policy
before attempting a neural implementation. A positive result would establish
system-level cumulative policy adaptation, not neural learning-to-learn or
general intelligence.

## Falsifiable hypothesis

After acquiring and independently verifying source procedures, a learner with
a reusable operator library will acquire more **functionally novel** target
procedures than:

1. a frozen induction policy;
2. an exact-procedure memory-only control; and
3. a sham-library control with matched persistent-state size and compute.

The target procedure's canonical AST hash and exhaustive truth-table hash must
be absent from every source procedure and every promoted operator record. A
target success therefore cannot be explained by direct retrieval of an earlier
answer.

The null interpretation is that any apparent gain is caused by direct memory
retrieval, source/target overlap, extra verifier calls, a larger candidate
budget, or an untracked increase in search computation.

## Closest prior work and novelty audit

| Prior work | What it established | Why this proposal is different |
|---|---|---|
| Lifecycle v0 | Engineered acquisition, verification, persistence, reset execution and externally triggered revision | v0 does not change its induction policy after verified experience |
| C1f–C1r | Engineered reuse, applicability, evidence control and safe adaptation in heterogeneous tasks | The series is not an end-to-end future-acquisition gate over the v0 procedure interface |
| Track B B0–B3 | One-shot neural proposal and functional novelty evaluation | B0–B3 has no source-to-target learning phase and B3 is now frozen as a negative baseline |
| C1m exact reuse adaptation | Exact certificate reuse and independently verified adaptation controls | C1m measures task-time reuse/cost, not improvement of the future induction policy |

This is not a B3 repair, a candidate-selection benchmark, or a larger-model
experiment. It tests a new causal variable: whether accepted prior procedures
change how later procedures are proposed.

## Scope and retained lifecycle

The following remain authoritative and unchanged:

- the allowlisted `digit_expr_mod10_v1` interpreter and canonical AST rules;
- independent deterministic verification;
- versioned persistent procedure records;
- context reset and post-reset execution;
- non-destructive invalidation and externally triggered revision;
- provenance and scaffolding accounting.

The adaptive component may change candidate ordering, motif selection or search
policy. It may not change verifier answers, promote its own proposal, bypass
the executor, rewrite the persistent procedure history, or detect a
contradiction without the registered external trigger.

## Procedure and operator representation

The manifest uses the existing allowlisted expression language. Every source
and target procedure is canonicalised and evaluated exhaustively over the 100
ordered digit pairs.

An operator-library record is not a stored answer procedure. It must contain:

- a parameterised AST motif with at least one typed hole;
- source evidence references and independent operator-validation references;
- an applicability declaration;
- a version, provenance and promotion status;
- no closed target AST, target truth table or raw target demonstrations.

An operator may be promoted only after it improves proposal/search outcomes on
predeclared operator-validation episodes that are separate from the final
target panel. Operator validation does not expose final target labels.

## Frozen task manifest requirements

The manifest must be generated and hashed before implementation tuning or
scientific execution. It must contain:

- 32 source episodes used for cumulative adaptation per seed;
- 144 final target episodes per seed, divided into two independently generated
  functionally novel target families of 72 episodes each;
- a 16-episode exact-replay audit panel per seed used only to verify the memory
  control;
- 32 operator-validation episodes per seed, allocated as 16 independent trials
  for each of at most two promotion-eligible operators;
- two registered seeds: 42 and 1337;
- disjoint canonical AST hashes and exhaustive truth-table hashes between
  source and final target procedures;
- balanced input coverage, operation depth and demonstration counts across
  arms;
- no reuse of the B3 scientific task manifest.

If the grammar cannot produce the required disjoint panels, the protocol is
invalid and must not be adapted after inspection of results.

The primary paired denominator is 288 task-seed units: 144 target tasks for
each of two seeds. The same task-seed unit is evaluated by every arm; source
and operator-validation records are not included in the target denominator.

The target task IDs are unique across seeds, and the frozen metadata audit
found zero cross-seed canonical-AST or truth-table intersections. The exact
dependence resolution and runner boundary are recorded in
`docs/CRUCIBLE_CUMULATIVE_ADAPTATION_PROTOCOL_INTEGRITY_AMENDMENT_20261028.md`.

## Experimental arms

All arms receive identical source demonstrations, target demonstrations,
candidate budgets, verifier-call caps, reset events and resource limits.

### A0 — frozen induction policy

The learner processes source episodes but does not retain a cross-episode
operator library or update its future proposal/search policy. It is the primary
future-acquisition baseline.

### A1 — exact procedure-memory control

The learner stores validated source procedures and may retrieve an exact
canonical procedure only when the current task matches an authorised source
identity. It cannot infer a new target procedure from an approximate or
functionally related match. The exact-replay audit panel must confirm that this
control can retrieve what it stores.

The final target panel is truth-table disjoint from source memory. Any target
hit is a leakage or implementation failure, not a success.

### A2 — sham-library control

The learner receives the same number, serialized size and access cost of
operator records as A3, but the records are deterministically permuted across
source families and cannot describe the target family. This controls for
persistent context, memory bandwidth and additional policy inputs.

### A3 — verified adaptive operator library

After each source procedure is independently verified, the learner proposes
parameterised operator motifs, tests them on the predeclared operator-validation
episodes, and promotes only validated motifs. The promoted library changes the
ordering or construction policy for later target proposals.

A3 receives no final target labels before each target proposal is fixed. Its
source adaptation cost is reported separately and cannot be hidden inside the
target inference budget.

## Exact budget matching

Every arm executes the same fixed accounting schedule for each target task:

- four candidate slots, including explicit null slots when an arm abstains;
- four operator-library lookup slots;
- four verifier calls, each over eight fixed independent examples;
- the same candidate-construction operation-unit cap;
- the same post-reset execution checks.

A0, A1 and A2 execute no-op or sham work in the slots that A3 uses for real
operator lookup. The counters are recorded per task and the run is invalid if
candidate slots, lookup slots, verifier calls or operation units differ across
arms. Wall-clock and CPU time are reported separately rather than treated as a
substitute for the matched operation budget.

## Primary outcome

For each final target episode, lifecycle success requires all of the following:

1. a proposal is emitted or the learner abstains;
2. the independent verifier accepts the proposal;
3. the canonical procedure is persisted with complete provenance;
4. the transient context is reset;
5. the persisted procedure executes correctly on fresh post-reset inputs;
6. no unsupported or contradictory procedure is promoted.

The primary metric is target lifecycle success rate. The primary contrast is:

`A3 target success − max(A0 target success, A2 target success)`

Exact-memory hits, target-label use, target truth-table overlap, verifier
budget violations and post-reset failures are reported separately and cannot be
counted as cumulative-learning success.

## Pre-registered success and failure criteria

The cumulative-adaptation hypothesis is supported only if, after both seeds
and both target families are fixed:

- A3 exceeds A0 by at least 10 percentage points on the pooled target panel;
- A3 also exceeds A2 by at least 10 percentage points;
- the direction of the effect is replicated for both seeds and both target
  families;
- the exact two-sided McNemar test on the 288 paired task-seed outcomes is
  recorded, and the paired 95% confidence interval from the preregistered
  20,000-sample stratified task-cluster bootstrap excludes zero. The bootstrap
  resamples complete task clusters within each seed-by-family stratum and never
  resamples arm outcomes independently;
- A3 has zero target truth-table or canonical-AST retrieval hits; and
- lifecycle verification, persistence, reset execution and revision gates are
  no worse than the registered baseline.

If A3 fails to improve over both controls, the mechanism is not supported under
the tested conditions. If an apparent gain comes from exact retrieval, target
overlap, extra verifier calls or unbalanced search, the result is invalid for
the cumulative-learning claim. If only one target family improves, the result
is supported but inconclusive and does not establish a reusable learning
strategy.

The 10-point threshold is a minimum scientific effect size, not a claim of
guaranteed power for every possible outcome distribution. The feasibility
preflight must report planning power under fixed discordant-pair scenarios
before execution.

If no eligible operator reaches the fixed 16/16 promotion requirement, A3 must
be recorded as `unactivated_inconclusive`. That outcome is not evidence against
an active learning policy, because no operator was activated. The promotion
threshold and validation allocation cannot be relaxed after outcomes are
observed.

## Resource and leakage controls

- Candidate counts, verifier calls and target proposal budgets are matched
  within the preregistered tolerance.
- Source adaptation cost, target inference cost and persistence size are
  reported separately.
- Target answers and final verifier outcomes remain inaccessible until the
  proposal is fixed.
- No target task is used for checkpoint selection, operator promotion or
  implementation tuning.
- The exact-memory and operator-library contents are hash-audited before final
  evaluation.
- The run stops on any source/target functional-hash collision or sealed-data
  access.
- No GPU allocation is made until this protocol passes a separate preflight
  and explicit execution approval.

## Scaffolding accounting

Retained scaffolding: verifier, executor, persistent store, reset, external
contradiction trigger, allowlisted grammar and exhaustive equivalence checker.

New tested component: verified operator library plus a policy adapter that uses
it to guide later proposal/search.

This first gate does **not** claim to reduce engineered scaffolding. It asks
whether verified experience can improve future acquisition while the existing
verification and execution anchors remain fixed. A later neural implementation
must be registered separately and cannot inherit a positive result from this
model-agnostic gate.

## Operator-promotion feasibility

The operator promotion rule is fixed as:

```text
LCB95(Beta(1 + successes, 1 + failures)) >= 0.80
```

With the registered Beta(1,1) prior, the minimum evidence is 16/16 successful
independent validation trials, whose lower 95% bound is approximately 0.80494.
The first gate therefore allocates 32 validation episodes to two eligible
operator candidates. A single failure prevents promotion; the threshold and
allocation cannot be relaxed after results are observed. Additional extracted
operators are recorded diagnostically but are not promotion-eligible in this
version.

## Frozen representative grammar pool

The runner uses the deterministic first-occurrence structural pool recorded at
`experiments/crucible_cumulative_adaptation_preflight_20261028/GRAMMAR_REPRESENTATIVE_POOL.jsonl`.
It contains 2,730 rows from 10,000 valid draws under the frozen depth-3,
node-7 grammar, with 692 functional representatives. This is not exhaustive;
unsampled expressions and functions are outside the evidence scope. Its exact
sampling method, exclusions and hash are frozen in the protocol-integrity
amendment.

## Runner construction boundary

The complete four-arm runner may be constructed and validated using synthetic
fixtures before execution approval. It must enforce sealed-target isolation,
matched budgets, independent verification, persistence/reset checks, replay and
resource accounting. A readiness report must distinguish implementation
readiness from scientific authorization. No target evaluation or sealed-answer
access is permitted during runner construction.

## Decision boundary

No scientific target evaluation, sealed-answer access, training run or GPU
allocation is approved by this document. The next allowed action is protocol
integrity amendment and synthetic runner validation. Scientific execution
requires a separate explicit approval after the readiness report passes.
