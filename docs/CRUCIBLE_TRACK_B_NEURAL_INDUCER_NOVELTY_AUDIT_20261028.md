# Track B neural procedure-proposer novelty audit

Date: 2026-10-28
Classification: **UNTESTED — proposal not approved**

## Proposed boundary

Replace only the deterministic procedure-induction component in Lifecycle v0:

`demonstrations → learned candidate proposal → deterministic verification → persistent procedure → reset → execution → contradiction/revision`

The proposer may emit candidate procedures, confidence and uncertainty. It may
not certify its own proposal, write an active procedure directly, or bypass
the independent verifier.

The required proposal/executor contract is defined in
`CRUCIBLE_TRACK_B_PROCEDURE_INTERFACE_DESIGN_20261028.md`. A rule-name-only
output is sufficient for Milestone 1 integration but is explicitly
insufficient for the Milestone 2 procedural-novelty claim.

Milestone 2 must report structural AST novelty separately from functional
novelty. Functional novelty is tested exhaustively over all 100 ordered pairs
of digits; equivalent truth tables do not count as new behaviour. The XOR
primitive is specifically defined as integer bitwise XOR followed by modulo 10.

## Closest previous work

1. **Lifecycle v0** — closest baseline. It uses a fixed eight-rule family,
   exact candidate filtering and deterministic probe selection. It proves the
   lifecycle contract but contains no learned proposer. The proposed boundary
   replaces one component while preserving the rest.
2. **C1f–C1h** — engineered sequential memory and safe-reuse controllers.
   They study reuse and applicability, but do not learn an executable
   procedure from demonstrations through a neural component.
3. **C1i–C1r** — learned or adaptive policies for applicability, evidence and
   intervention. They do not implement the Lifecycle v0 proposer/verifier/
   persistence interface and therefore are supporting precedents, not the same
   experiment.
4. **Track A Phase VII** — neural shared-latent relational transfer. It is a
   different task and has no persistent procedure lifecycle, reset gate or
   verifier boundary.

## Novelty judgement

The proposed component replacement is not a duplicate of the indexed work,
provided it keeps the Lifecycle v0 interface and does not add neural
contradiction detection, neural execution or learned persistence at the same
time. The first version would demonstrate neural-assisted proposal under a
bounded known hypothesis family, not general unfamiliar-procedure acquisition.

## Required predictions before approval

- The learned proposer must preserve every v0 lifecycle gate at matched task,
  seed and verifier conditions.
- Incorrect or uncertain proposals must be rejected or abstained on by the
  verifier; proposer confidence cannot promote a procedure.
- The scaffolding budget must show which component was removed and whether
  oracle calls, evaluator intervention or task-specific rules increased.
- A fixed-family success must be labelled **neural-assisted bounded proposal**.
  A stronger claim requires held-out procedure families or a candidate grammar
  not exposed as the answer set during training.
- A Milestone 2 success requires both an unseen canonical AST and an unseen
  exhaustive truth table; syntactic rearrangement of a known function is not
  procedural novelty.
- Failure of the proposer must not be interpreted as failure of the retained
  verifier, persistence, executor or revision components.

## Approval status

No training run, architecture change or GPU allocation is approved by this
memo. The next permitted action is a protocol and leakage review against the
registered baseline, scaffolding budget and procedure-interface design.
