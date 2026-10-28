# Executable operator pathway diagnostic

This is a bounded, synthetic CPU diagnostic. It does not rerun the 288-task
benchmark, open target answers, or allocate GPU compute.

## Result

The complete pathway is functional:

1. Three independently verified source procedures are reduced to a canonical
   `add_mod10` AST motif with two typed `expr_mod10` holes.
2. The motif is instantiated against 16 independently generated applicable
   validation tasks.
3. All 16 candidates execute correctly and pass independent verification.
4. The operator is promoted.
5. Promotion changes future candidate construction: the active path constructs
   `add_mod10(b, 3)`, which is absent from the frozen fallback pool containing
   only a `max(a,b)` distractor.
6. The active and frozen proposal orders differ.

The accelerated CPU reservoir was exercised with the promoted `OperatorRecord`
itself; its first candidate and complete proposal order matched the parent
implementation. The 16 synthetic validation tasks use distinct deterministic
demonstration and independent-verification draws.

## Failure attribution

Synthetic negative controls reach the intended stages independently:

| Control | Failure stage |
| --- | --- |
| Applicability mismatch | `applicability_matching` |
| Empty typed-hole vocabulary | `candidate_construction` |
| Wrong binding/output | `incorrect_output` |
| Wrong independent verification examples | `verification` |

The diagnostic also passes deterministic replay. The full machine-readable
trace, operator record, hashes and replay digest are in `PATHWAY_REPORT.json`
and `HASHES.json`.

The existing preflight validation manifest was audited separately but was not
rerun as a benchmark: `min` produced 0/16 and `add_mod10` had only 10 allocated
trials, 0/10. Those records remain unpromoted, so a future benchmark run needs
a regenerated operator-specific manifest with 16 applicable tasks per
operator before it can evaluate the repaired mechanism.

## Interpretation

This changes the implementation behavior from recording a root signature and
optionally applying a root-name ranking bonus to using an executable,
canonical typed-hole motif to construct actual future candidates. It does not
demonstrate neural learning or generalization: the synthetic operator and
grammar are engineered, and the registered benchmark was intentionally not
rerun.
