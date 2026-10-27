# Crucible cumulative-learning research proposal

Date: 2026-10-28
Status: **proposal only — no implementation, training or execution approved**

## Executive decision

The strongest next mechanism is a **verified operator library with an
operator-guided proposal policy**. It is a small, auditable form of cumulative
adaptation: verified source procedures are abstracted into parameterised
motifs, tested independently, and used to change the search policy for later
procedures.

This is deliberately not a B3 repair and not another neural architecture
sweep. The first implementation should be model-agnostic and CPU-only. It
tests whether verified experience can improve future acquisition at all. A
neural policy adapter is a later, separate question.

## Ledger audit

The relevant indexed evidence is:

| Ledger record | Status | What it establishes | Remaining gap |
|---|---|---|---|
| `learning-lifecycle-v0-engineered-baseline` | ESTABLISHED | Acquisition, verification, persistence, reset execution and externally triggered revision work in the bounded v0 family | The induction policy does not improve from prior procedures; contradiction detection is externally invoked |
| `track-b-c1-engineered-controller-series-v1` | INCONCLUSIVE | Engineered reuse, applicability, active evidence, counterexample refinement and safe-reuse controls have been tested | No single end-to-end future-acquisition gate and no neural procedure proposer |
| `c1m-exact-reuse-adaptation` | indexed under the C1 series | Exact certificate reuse can reduce task cost; broad reuse can cause harmful transfer | Reuse does not demonstrate that the future induction policy learned a reusable strategy |
| `track-b-neural-proposer-protocol-v1` | FAILED under tested conditions | B3 reached training target coverage but acquired 0/48 held-out novel procedures; lifecycle anchors remained functional | The tested B3 configuration is retired; cumulative adaptation was not tested |
| `track-b-procedure-interface-design-v1` | Design boundary | Canonical ASTs, exhaustive functional novelty and deterministic verification are available as contracts | No adaptive operator library or future-acquisition policy has been implemented |
| `future-500m-cross-domain-reasoning-curriculum-v1` | UNTESTED | A future curriculum has been recorded | No evidence that cross-domain data improves the learning mechanism |

The closest prior experiments therefore cover memory reuse, evidence policy and
one-shot proposal, but none tests whether accepted experience changes the
learner's subsequent acquisition behaviour while direct retrieval is blocked.

## Candidate mechanisms considered

| Mechanism | Potential benefit | Main limitation in Crucible | Decision |
|---|---|---|---|
| Exact episodic procedure memory | Immediate reuse of a verified solution | Cannot distinguish remembering a solution from learning a reusable acquisition strategy | Retain as A1 memory-only control |
| Neural fast weights or meta-learning | Can adapt a proposal policy from episodes | Requires a competent base proposer, introduces optimizer/architecture confounds, and B3 did not establish that competence | Defer until the mechanism-level gate passes |
| Continual fine-tuning of the proposer | Directly changes future proposal behaviour | Risks catastrophic forgetting, target leakage and ambiguous attribution between more training and better learning | Defer; later neural implementation only |
| Active evidence-request policy | Can reduce demonstrations needed to identify a procedure | Improves evidence efficiency, not necessarily reusable procedure construction | Retain as a measured secondary cost, not the primary mechanism |
| Bayesian/MDL abstraction library | Gives uncertainty, provenance and conservative promotion | Can become an engineered catalogue of shortcuts unless target retrieval is excluded | Use as the auditable policy implementation |
| Program-synthesis search prior | Reusable motifs can guide construction of new ASTs | Can merely reorder an exhaustive search and claim false abstraction | Use only with matched candidate/verifier budgets and sham-library controls |

The selected mechanism combines the last two entries. It is architecturally
plausible because the existing verifier and AST interpreter already define a
safe boundary, and it is meaningfully distinct from prior work because its
success criterion is improvement on later, functionally novel tasks—not reuse
of a stored answer.

## Proposed internal state

The adaptive learner maintains five explicitly versioned state components:

```text
P_t  verified procedure records retained by Lifecycle v0
O_t  verified operator records, each a parameterised AST motif with typed holes
Q_t  proposal-policy statistics for grammar expansion actions
C_t  provenance and confidence certificates for each operator promotion
B_t  per-episode candidate, verifier and compute counters
```

`P_t` remains the ordinary v0 procedure store. `O_t` is not allowed to contain
a closed target AST, a target truth table or raw target demonstrations. `Q_t`
only changes proposal ordering and candidate construction; it cannot alter the
verifier, executor or persistence rules.

An operator record contains:

```text
operator_id, canonical_motif, typed_holes, source_refs,
validation_refs, applicability, successes, trials, status, provenance
```

The record is promoted only after independent operator-validation episodes
pass and the operator's structural and functional hashes are audited against
the final target manifest.

## Proposed update rule

For each operator `o`, maintain a Beta-Bernoulli transfer estimate:

```text
success_o, trials_o
posterior_o = Beta(1 + success_o, 1 + trials_o)
```

After a source procedure is independently verified, deterministic motif
extraction proposes operator candidates. Each candidate is tested on the
predeclared operator-validation episodes. The library promotes `o` only if:

```text
LCB95(posterior_o) >= 0.80
```

and no source/target functional-hash collision exists.

For a later candidate-construction action `e`, the adaptive policy uses:

```text
score_t(e) = base_score(e)
             + 0.25 * mean(LCB95(o) for operators matching e)
             - 0.01 * estimated_cost(e)
```

The policy changes only the order in which candidates are generated. The
candidate budget and verifier-call cap are fixed across arms. Failed transfer
updates the relevant operator statistics and can downgrade an operator; no
operator is silently deleted.

This is an intentionally conservative, interpretable update rule. It tests
whether validated abstractions improve future search without pretending that a
statistical policy update is already neural meta-learning.

## Verification boundary and persistence

The independent verifier remains authoritative:

1. the adaptive policy proposes an AST or abstains;
2. the verifier canonicalises and executes it on independent examples;
3. only the verifier can promote the procedure to `P_t`;
4. the operator library receives only verified source outcomes and its own
   predeclared validation outcomes;
5. target labels remain inaccessible until each target proposal is fixed;
6. reset clears transient hypotheses but retains versioned `P_t`, `O_t` and
   provenance;
7. externally triggered contradiction invalidates the active procedure without
   destructive overwrite, as in Lifecycle v0.

The operator library must be serialised separately from procedure memory so a
future evaluation can distinguish:

- direct procedure retrieval;
- operator-guided construction;
- frozen-policy induction; and
- sham-memory effects.

## Decisive experiment

The full preregistered design is in
`CRUCIBLE_CUMULATIVE_ADAPTATION_GATE_PROTOCOL_20261028.md` and its JSON
manifest. Its four arms are:

- **A0 frozen induction policy** — no cross-task operator update;
- **A1 exact procedure memory** — direct retrieval only, with an exact-replay
  audit panel;
- **A2 sham operator library** — matched persistent state and access cost but
  irrelevant operator assignments;
- **A3 verified adaptive operator library** — source procedures are abstracted,
  independently validated and used to guide later proposal/search.

The proposed manifest has 32 source episodes, 48 final target episodes in two
target families, 16 exact-replay audit episodes and seeds 42/1337. Source and
target partitions must be disjoint by both canonical AST hash and exhaustive
100-row truth-table hash. The B3 manifest is not reused.

The primary metric is complete target lifecycle success:

```text
proposal → independent verification → persistence → reset → fresh execution
```

The primary contrast is:

```text
A3 target success − max(A0 target success, A2 target success)
```

Support requires at least a 10-point pooled improvement over both A0 and A2,
replication in both seeds and target families, a paired 95% interval excluding
zero, zero target retrieval/overlap hits, and preserved lifecycle gates. A1's
exact-replay audit must pass, but A1 success on novel targets is not counted as
learning.

## Implementation requirements

Before any scientific run, the implementation must provide:

1. a deterministic grammar enumerator and canonical AST interpreter;
2. an independent truth-table and hash-equivalence auditor;
3. source/operator-validation/target manifest generation with frozen hashes;
4. the four arm implementations with identical task and budget interfaces;
5. operator extraction, Beta statistics, lower-confidence promotion and
   downgrade logic;
6. separate persistent stores for procedures and operators;
7. target-label and sealed-test access guards;
8. per-task candidate, verifier, CPU/GPU, memory and lifecycle counters;
9. deterministic replay from saved manifests and state hashes;
10. a validator that stops on overlap, budget mismatch, leakage or lifecycle
    failure.

No neural component is required for this first mechanism gate. If it passes,
the next proposal may replace only `Q_t` with a neural policy adapter while
keeping `O_t`, the verifier and the executor fixed.

## Estimated compute and storage

These are planning estimates, not measured results:

| Work | Expected resource |
|---|---|
| Grammar enumeration, truth tables and leakage audit | under 1 CPU-hour; under 100 MB |
| Four-arm deterministic preflight and replay | under 1 CPU-hour; under 200 MB |
| Full 4-arm × 2-seed CPU gate | roughly 2–8 CPU-hours, depending on verifier implementation |
| GPU requirement | none for the proposed first gate |
| Persistent artifacts | expected below 1 GB including traces and hashes |

The estimates must be benchmarked during preflight. No GPU rental or cloud
allocation is justified before the CPU feasibility gate passes.

## Major risks and explicit failure criteria

| Risk | Detection | Classification |
|---|---|---|
| Operator record encodes a target answer | AST/truth-table overlap audit | Invalid |
| A3 receives more search or verifier budget | Per-arm counter comparison | Invalid |
| Gain is only exact retrieval | A1 replay/target-hit breakdown | Memory transfer, not learning |
| Gain appears in one target family only | Family-stratified result | Supported but inconclusive |
| A3 does not beat A0 and A2 | Pre-registered primary contrast | Falsified under tested conditions |
| Lifecycle breaks after adaptation | Verification/reset/interference gates | Lifecycle failure; no adaptation claim |
| Policy update helps only source-like tasks | Target-family and identifier remapping | No reusable strategy established |

## Recommended next implementation step

Implement only the **CPU feasibility and leakage preflight**:

1. enumerate the expression grammar and construct the source/operator-validation/
   target manifest;
2. verify structural and exhaustive functional disjointness;
3. implement the independent operator record schema and hash auditor;
4. run the update rule on synthetic training-only fixtures;
5. produce a cost estimate and runner-validation report.

Do not train a neural proposer, run the four-arm scientific gate, access final
target labels or reopen Track B until this preflight has been separately
reviewed and explicitly approved.

## Decision boundary

This proposal does not claim that cumulative learning has been demonstrated.
It provides one falsifiable, low-cost mechanism test that is distinct from
Track B B3 and from exact procedure memory. A positive result would justify a
later neural policy-adapter proposal; a negative result would retire this
operator-guided mechanism before more architecture or dataset expansion.
