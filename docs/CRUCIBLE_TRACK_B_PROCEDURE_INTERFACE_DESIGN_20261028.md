# Crucible Track B procedure interface design

Date: 2026-10-28
Status: design only; untested; no training approved

## Purpose

Define the boundary between a learned procedure proposer and the deterministic
verifier/executor before any neural-assisted Lifecycle experiment is approved.
The interface must support the two intended milestones without quietly moving
the hard-coded procedure vocabulary into a new component.

## Two milestones

### Milestone 1 — Neural integration

The proposer may select or rank one of the existing eight v0 procedures. The
verifier, persistence store, reset, executor and revision path remain
unchanged. This is a bounded integration test and must be reported as
**neural-assisted bounded proposal**.

### Milestone 2 — Procedural novelty

The proposer emits a canonical compositional procedure representation. The
procedure may be a new composition of allowed primitives rather than a named
member of the v0 eight-rule list. The verifier must accept or reject it from
independent evidence, and the persistent executor must run the stored
representation after reset.

Milestone 2 requires procedure families or expression structures withheld from
proposer training. A new identifier for one of the eight existing rules does
not qualify as procedural novelty.

## Canonical proposal schema

The proposed interface is a data contract, not an implementation claim:

```json
{
  "schema_version": "crucible.procedure.v1",
  "language": "digit_expr_mod10_v1",
  "ast": {
    "op": "add_mod10",
    "args": [
      {"op": "var", "name": "a"},
      {"op": "var", "name": "b"}
    ]
  },
  "input_signature": ["a", "b"],
  "output_type": "digit_mod10",
  "applicability": {"domain": "digits_0_9_binary"},
  "confidence": 0.0,
  "uncertainty": {"kind": "candidate_set", "alternatives": []},
  "evidence_refs": [],
  "provenance": {}
}
```

The proposer returns one or more proposals plus uncertainty. It never writes
an active procedure directly and its confidence never overrides verification.
The verifier canonicalises the AST and computes the procedure hash; proposer
supplied hashes are not trusted.

## Allowlisted expression language

The first executable language should be deliberately small and total:

- variables: `a`, `b`;
- constants: integers `0..9`;
- primitive operations: `add_mod10`, `sub_mod10`, `mul_mod10`, `xor_mod10`,
  `min`, `max`;
- structural limits: maximum depth and node count fixed in the protocol;
- no Python, shell, arbitrary function names, memory reads or external calls;
- deterministic evaluation with explicit domain and output-type checks.

`FIRST` and `SECOND` may be represented by `var(a)` and `var(b)` in the
canonical language. Named v0 rules can be imported through a compatibility
adapter, but the persisted canonical form should be the AST.

## Component contract

| Component | Milestone 1 | Milestone 2 |
|---|---|---|
| Proposer | predicts/ranks known rule candidates | emits AST candidates and uncertainty |
| Evidence controller | existing bounded probe policy | separately measured; not silently expanded |
| Verifier | deterministic independent checks | same verifier over canonical AST |
| Executor | v0 rule adapter | allowlisted AST interpreter |
| Persistence | versioned rule metadata | versioned canonical AST, language and hash |
| Reset | clear transient hypotheses | same |
| Revision | existing invalidation/relearn path | same until contradiction detection is separately replaced |

## Verification and leakage rules

1. Demonstrations available before proposal are recorded with provenance.
2. The proposer cannot read evaluator labels, held-out answers, the persistent
   store's active rule or verifier outcomes from the current task before it
   emits its proposal.
3. Verification examples are independent and remain evaluator-only until the
   proposal is fixed.
4. Promotion requires schema validation, deterministic execution and complete
   verification success.
5. Multiple surviving procedures remain uncertain; the system must not promote
   on confidence alone.
6. The verifier may retain the authoritative answer oracle in the first
   integration, but all oracle calls and intervention points are counted in
   the scaffolding budget.

## Persistence contract

The active record should persist:

- `language` and canonical AST;
- canonical procedure hash;
- applicability and type/domain declarations;
- confidence and uncertainty at promotion time;
- demonstration, probe and verification references;
- provenance and version history;
- invalidation events without destructive overwrite.

Raw demonstrations remain outside the procedure record unless a separate
retention experiment explicitly permits them. Reloading the store must be
sufficient for deterministic execution of the promoted AST.

## Approval gates

Before training, a protocol must specify:

- the exact proposer input and output tensors/records;
- fixed-family Milestone 1 and held-out-expression Milestone 2 splits;
- matched v0 seeds, verifier, query counts and reset semantics;
- a leakage audit for demonstrations, verification labels and persistent
  state;
- lifecycle gates and negative controls;
- the scaffolding-budget delta;
- failure attribution between proposal, verification, persistence, execution
  and revision.

No result from this design document establishes that a neural model can emit,
verify, persist or execute compositional procedures.
