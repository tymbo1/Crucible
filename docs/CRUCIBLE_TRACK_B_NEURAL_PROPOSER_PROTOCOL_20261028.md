# Crucible Track B neural procedure-proposer protocol v1

Date: 2026-10-28
Status: preregistered design only; no run approved

## Objective

Test whether a learned procedure proposer contributes beyond deterministic
program search while preserving the complete Lifecycle v0 contract:

`propose → verify → persist → reset → execute → process contradiction → revise`.

The experiment separates neural integration from procedural novelty. A fixed
eight-rule result is Milestone 1. A verified, persisted procedure with a new
function and an unseen compositional structure is Milestone 2.

## Primary hypotheses

- **H1 — Lifecycle preservation:** A learned proposer can replace deterministic
  candidate induction without reducing the v0 lifecycle gates.
- **H2 — Neural contribution:** At matched grammar, verifier, task and compute
  budgets, the learned proposer improves held-out functional acquisition or
  reduces search/verifier cost relative to deterministic AST search.
- **H3 — Procedural novelty:** A learned proposer can acquire a functionally
  novel procedure from demonstrations, verify it independently, persist it and
  execute it after reset.

Failure of H2 or H3 retires that tested proposer/configuration; it does not
imply that neural program induction is impossible in general.

## Arms

All arms use the same task manifests, verifier, executor, persistence schema,
reset semantics, contradiction fixture and evaluation queries.

| Arm | Induction mechanism | Purpose |
|---|---|---|
| B0 | Lifecycle v0 deterministic eight-rule filter | Positive lifecycle reference |
| B1 | Deterministic bounded AST enumeration with consistency/complexity ranking | Non-neural program-search baseline |
| B2 | Learned proposer scoring the fixed eight-rule catalogue | Milestone 1 neural integration |
| B3 | Grammar-constrained learned AST decoder that emits candidate ASTs directly | Milestone 2 neural-assisted proposal |

B1 and B3 receive the same grammar, maximum AST depth/node limits and output
candidate budget. B2 is not allowed to be interpreted as procedural novelty.

### Candidate-source requirement

B3's primary implementation must generate serialized ASTs directly under the
grammar. It must not receive an exhaustive candidate list containing the target
procedure and then merely rerank that list. A catalog-reranking implementation
may be run only as a separately labelled diagnostic (`B3-R`) and cannot support
the independent-proposal or procedural-novelty claims.

Every result records `candidate_source` as one of `deterministic_enum`,
`neural_generated` or `catalog_ranked`. Duplicate canonical ASTs count once
against the proposal budget.

## Procedure language

The executable grammar is the registered `digit_expr_mod10_v1` language:

- variables `a`, `b`;
- constants `0..9`;
- `add_mod10`, `sub_mod10`, `mul_mod10`, `xor_bits_mod10`, `min`, `max`;
- fixed maximum depth and node count;
- deterministic total execution over the 100 ordered digit pairs;
- no arbitrary code, external calls, memory reads or unregistered operators.

`xor_bits_mod10(a,b)` means `(integer_bitwise_xor(a,b)) mod 10`.
Canonical AST normalization and exhaustive functional-equivalence rules are
defined in the procedure-interface design.

## Task families and splits

The manifest must be generated and hashed before training. It must contain
three disjoint task strata:

1. **Fixed-family integration:** hidden procedures are the eight v0 rules;
   demonstrations are new, and the verifier/query pairs are disjoint.
2. **Structural novelty:** target ASTs use grammar structures absent from
   proposer training, while their functions are checked against all training
   truth tables.
3. **Functional novelty:** target truth tables differ from every proposer
   training procedure. Target ASTs must also be structurally unseen for a full
   Milestone 2 claim.

For every task, use disjoint sets for demonstrations, independent verification
and post-reset queries. The initial digit domain has 100 ordered pairs, so the
manifest must reserve all remaining pairs after demonstrations and verification
for post-reset scoring where possible.

Training tasks that are not uniquely verifiable under the fixed verifier budget
may be excluded before optimization, but the manifest must report their count
and proportion. Validation and test tasks are not silently discarded. They are
labelled `identifiable_under_budget` or `unresolvable_under_budget` and remain
in the all-task denominator. Report both conditional acquisition on identifiable
tasks and abstention precision/coverage on unresolvable tasks.

The learned proposer is trained only on training-family tasks. Validation and
test families, AST hashes and truth-table hashes remain unavailable to proposer
training, checkpoint selection and persistent state.

## Common lifecycle gates

Every arm must report:

1. proposal status and uncertainty;
2. verifier acceptance/rejection and independent verification accuracy;
3. canonical procedure hash and provenance completeness;
4. persistence reload after transient context is cleared;
5. post-reset execution accuracy and coverage;
6. contradiction handling using the common externally supplied contradiction
   fixture, including invalidation and versioned relearning;
7. no-persistence, raw-example, insufficient-evidence and contradictory-input
   negative controls;
8. interference on procedures retained from earlier tasks.

The first protocol does not replace contradiction detection. It measures
revision after a common contradiction event; independent contradiction
detection is a later boundary.

## Primary metrics

### Lifecycle reliability

- task acquisition rate;
- independent verification pass rate;
- post-reset exact accuracy and coverage;
- contradiction invalidation and validated successor rate;
- provenance/persistence integrity;
- negative-control failure rate;
- retained-procedure interference.

### Neural contribution

- functional acquisition on held-out families;
- structural and functional novelty rates;
- number of candidate evaluations;
- verifier calls and active probes;
- wall-clock and CPU/GPU compute;
- proposal rank and uncertainty calibration.

Also report, separately for every task and arm:

- candidate evaluations and candidate-output count;
- verifier calls and oracle calls;
- proposer forward passes and decoder tokens;
- CPU seconds, GPU seconds, wall-clock seconds and peak memory;
- training compute and inference compute separately.

The primary comparison is B3 versus B1 on held-out functionally novel tasks at
matched candidate/verifier budgets and also reports a matched-total-compute
comparison. Candidate counts are not treated as a proxy for compute. A neural
advantage requires either higher lifecycle-preserving functional acquisition at
matched total compute or equal acquisition with lower measured total compute
without increased oracle/intervention cost. Matching B1 does not establish a
neural advantage.

## Scaffolding accounting

For every arm record:

- hard-coded primitive count and grammar limits;
- named procedure vocabulary available to the proposer;
- candidate-generation and ranking logic;
- oracle calls and verifier calls;
- externally triggered intervention points;
- evaluator-only information;
- learned parameters and training labels;
- persistent state contents.

No claim of reduced scaffolding is valid if removing candidate filtering is
offset by an unreported increase in oracle access, candidate generation,
privileged labels or evaluator intervention.

## Leakage controls

- no held-out answers, truth tables or target ASTs enter proposer inputs;
- verifier outputs are unavailable until after proposal emission;
- persistent state contains only promoted canonical procedures and provenance;
- AST and truth-table hash partitions are checked before execution;
- no raw-example retrieval may be used as a procedure substitute;
- all seeds, manifests, grammar limits and source hashes are frozen before
  training.

The preflight must reject any implementation that cannot expose these separate
candidate, verifier and device-compute counters.

## Compute budget

This protocol does not authorize execution. If approved after preflight:

- manifest generation and exhaustive truth-table checks: CPU-only;
- B1 search budget: fixed candidate count and verifier-call cap per task;
- learned proposer: one small fixed architecture, two seeds, maximum 2,000
  optimizer updates per seed;
- preflight: 50 updates maximum and no scientific checkpoint selection;
- full run: maximum two GPU-hours total, excluding CPU manifest generation;
- stop immediately on leakage, competence-gate or reproducibility failure.

## Decision rules

- **Established:** all lifecycle gates pass and the narrow preregistered claim
  is supported.
- **Supported but inconclusive:** informative result with unresolved confound;
  no follow-up is automatic.
- **Falsified under tested conditions:** preregistered prediction fails with
  valid competence and leakage controls.
- **Invalid:** protocol, leakage, verifier, persistence or competence failure
  prevents interpretation.
- **Duplicate:** the proposed comparison adds no material evidence beyond a
  registered prior result.

The first approval review must confirm the manifest, exact proposer interface,
candidate budget, AST equivalence audit, leakage audit and compute estimate.
It must also confirm the direct-generation provenance of B3, the excluded-task
proportion and the all-task abstention analysis. No neural run is approved by
this document alone.
