# Crucible operator-activation diagnostic

Date: 2026-10-28
Classification: bounded CPU diagnostic
Benchmark rerun: no
Target answers opened: no
Target lifecycles executed: no

## Finding

The saved A3 run did not reach operator construction or verification. The
current implementation selects `validation_rows[:16]` as one shared mixed
prefix for every eligible operator, then counts only root-signature equality.
That makes the promotion gate an applicability-label check rather than a
verified operator test.

The current code does contain a narrow downstream hook: once a record is
marked promoted, `candidate_pool_for_task` gives matching candidate roots a
`+1` ranking bonus. A synthetic 16/16 verified `add_mod10` operator changed a
two-candidate order from `max → add_mod10` to `add_mod10 → max`. Thus the hook
is functional, but the intended motif-to-candidate construction path is not.

## Eligible operators from the saved traces

The runner’s existing rule selects the first two distinct source root
signatures: `min` and `add_mod10`.

### `min`

Diagnostic motif:

```json
{"op":"min","args":[{"op":"hole","id":"arg0","type":"expr_mod10"},{"op":"hole","id":"arg1","type":"expr_mod10"}]}
```

Typed holes are positional expression holes `arg0` and `arg1`. This motif is
diagnostic metadata only; the current `OperatorRecord` does not store it.

Source procedures:

`source_s42_000`, `source_s42_002`, `source_s42_003`, `source_s42_010`,
`source_s42_011`, `source_s42_014`, `source_s42_028`, `source_s1337_000`,
`source_s1337_002`, `source_s1337_003`, `source_s1337_010`,
`source_s1337_011`, `source_s1337_014`, `source_s1337_028`.

Current validation prefix: **3/16** root-signature matches.

Successes: `operator_validation_s42_000`, `_009`, `_010`.
Failures: `_001`, `_002`, `_003`, `_004`, `_005`, `_006`, `_007`, `_008`,
`_011`, `_012`, `_013`, `_014`, `_015`.

### `add_mod10`

Diagnostic motif:

```json
{"op":"add_mod10","args":[{"op":"hole","id":"arg0","type":"expr_mod10"},{"op":"hole","id":"arg1","type":"expr_mod10"}]}
```

Typed holes are again positional expression holes `arg0` and `arg1`; the
current baseline record stores neither the motif nor the holes.

Source procedures:

`source_s42_001`, `source_s42_006`, `source_s42_012`, `source_s42_015`,
`source_s42_021`, `source_s42_022`, `source_s42_027`, `source_s42_030`,
`source_s1337_001`, `source_s1337_006`, `source_s1337_012`,
`source_s1337_015`, `source_s1337_021`, `source_s1337_022`,
`source_s1337_027`, `source_s1337_030`.

Current validation prefix: **2/16** root-signature matches.

Successes: `operator_validation_s42_006`, `_008`.
Failures: `_000`, `_001`, `_002`, `_003`, `_004`, `_005`, `_007`, `_009`,
`_010`, `_011`, `_012`, `_013`, `_014`, `_015`.

## Failure-stage accounting

For `min`, all 13 failures are applicability mismatches. For `add_mod10`,
all 14 failures are applicability mismatches. Candidate construction,
incorrect output and independent verification each record zero observed
failures because those stages are not implemented in the current validation
path and are never reached.

On the first four saved target-input rows, forcing either record to promoted
changed the candidate order on 3/4 rows. This is a ranking-only diagnostic;
it is not a target evaluation.

## Synthetic activation proof

A synthetic source procedure and 16 matching validation procedures were
independently verified. The `add_mod10` record reached 16/16, crossed the
fixed lower-bound gate, and became `promoted`. With an equal-size `max`
distractor and no target labels:

- A0 order: `max`, then `add_mod10`.
- Activated A3 order: `add_mod10`, then `max`.

This demonstrates that the existing downstream policy hook can alter proposal
order when activation is supplied.

## Next engineering action

Before any new benchmark, replace the root-signature stub with operator
records containing canonical motifs and typed holes, allocate exactly 16
operator-specific validation trials per eligible operator, and trace motif
instantiation, candidate construction, output correctness and independent
verification separately. Pass synthetic 16/16 activation, rejection,
incorrect-output and deterministic-order fixtures before freezing a new
manifest.

Full machine-readable traces and hashes are in `DIAGNOSTIC_REPORT.json`,
`VALIDATION_TRACE.jsonl`, `SYNTHETIC_ACTIVATION.json` and `HASHES.json`.
