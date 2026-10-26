# Crucible Learning Lifecycle v0

Date: 2026-10-25  
Status: design frozen before benchmark execution  
Implementation: deterministic CPU prototype

## Objective

Demonstrate the smallest auditable learning lifecycle:

`demonstrations → candidate hypotheses → evidence acquisition → verification → persistent procedure → context reset → execution → contradiction revision`.

This is an engineered procedural learner, not a neural or autonomous-intelligence claim.

## Bounded hypothesis space

The learner receives examples for an undisclosed binary operation over digits `0..9` modulo 10. The candidate family is fixed but the task alias and target rule are not disclosed:

`ADD_MOD10`, `SUB_MOD10`, `MUL_MOD10`, `FIRST`, `SECOND`, `XOR_MOD10`, `MAX`, `MIN`.

Each candidate is an executable procedure with a canonical signature. Candidate enumeration is deterministic.

## Learning and verification

1. Filter candidates against demonstrations.
2. If multiple candidates remain, request bounded probes at inputs where candidates disagree.
3. If no candidates remain, record a failed/contradictory strategy without promotion.
4. When one candidate remains, test it on independent verification examples.
5. Promote only after verification succeeds; persist the procedure signature, applicability, evidence references, confidence, provenance and version.

The learner stores evidence references and verification outcomes, not an answer table used as a substitute for execution.

## Persistence and revision

Temporary hypotheses and demonstrations are discarded on context reset. The persistent store retains versioned procedure metadata and revision history.

If later evidence contradicts a validated procedure, the active version is invalidated, the contradiction is preserved, and a new learning attempt may create a successor version. No destructive overwrite is permitted.

## Controls

- full lifecycle with procedure persistence;
- persistence disabled;
- raw-example retrieval without a procedure;
- insufficient demonstrations with no informative probe budget;
- contradictory demonstrations;
- stale/contradictory evidence after initial validation.

## Metrics

- acquisition and verification success;
- demonstrations and probes used;
- post-reset execution accuracy and coverage;
- uncertainty/abstention rate;
- procedure revisions and invalidations;
- interference on previously retained procedures;
- persistence integrity and provenance completeness;
- negative-control behavior;
- deterministic compute cost.

## Interpretation boundary

Success demonstrates that an explicit bounded learner can complete and retain a procedural-learning lifecycle. It does not demonstrate neural learning, autonomous hypothesis-space discovery, learning-to-learn, or general intelligence.
