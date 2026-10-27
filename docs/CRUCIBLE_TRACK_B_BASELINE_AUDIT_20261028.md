# Crucible Track B baseline audit

Date: 2026-10-28
Status: read-only repository audit; no training or experiment execution

## Purpose

Re-establish the original Track B objective before approving a neural
replacement: acquire an unfamiliar procedure, verify it, persist it, survive a
temporary-context reset, execute it, respond to contradiction and revise it.

## Narrowly established baseline

`crucible_learning_lifecycle_v0_20251025` is a valid engineered positive
control under its documented conditions:

| Gate | Result |
|---|---:|
| Procedures acquired and independently verified | 24/24 |
| Post-reset executions | 288/288 |
| Retained procedure aliases | 24 |
| Insufficient-evidence tasks left uncertain | 18/24 |
| Contradictory demonstrations promoted | 0/24 |
| Raw-example retrieval after novel queries | 0/288 |
| No-persistence post-reset coverage | 0/288 |
| Revision control | version 1 invalidated, version 2 validated |

The four unit tests pass. This establishes an auditable bounded symbolic
lifecycle, not neural learning, autonomous hypothesis-space discovery or
general intelligence.

## Engineered dependency boundary

The v0 learner currently supplies the rule family, exact candidate filtering,
disagreement-based probe selection, executable procedures and persistent
versioning. An external oracle supplies probe and verification outputs. The
executor is deterministic. Contradiction handling is not independently
detected: the benchmark explicitly calls `learner.invalidate(...)` before
relearning. The complete accounting is in
`CRUCIBLE_TRACK_B_SCAFFOLDING_BUDGET_20261028.json`.

## Related C1 evidence

The C1f–C1r series contains useful engineered controller work on reuse,
applicability, abstention, active evidence, counterexample refinement,
corroboration and intervention testing. It does not demonstrate a neural
model acquiring a new executable procedure and retaining it across reset.
The series is registered as a separate heterogeneous evidence record rather
than being conflated with Lifecycle v0.

The 170M persistent-memory runs exercised scratchpad reuse, but their short
budgets did not establish a learning-lifecycle improvement and the skip
intervention was not active in the 500-step run.

## Provenance correction

The former coverage audit searched only for `REPORT_FINAL.md`. Lifecycle v0 is
reported in `REPORT.md`, so it was absent from the claimed 40-report index.
Coverage discovery now searches both canonical report names. The C1 summaries
are also linked explicitly as artifacts; they are not silently treated as
neural lifecycle evidence.

## Decision

Track A remains frozen at Phase VII. Track B v0 is the preserved positive
control. No neural training is approved by this audit.

The first candidate boundary is a learned procedure proposer constrained by
the existing verifier, persistence, reset, executor and revision machinery.
That is an integration test over the fixed eight-rule family. It must not be
described as unfamiliar-procedure discovery until evaluation includes
procedure families withheld from the proposer.
