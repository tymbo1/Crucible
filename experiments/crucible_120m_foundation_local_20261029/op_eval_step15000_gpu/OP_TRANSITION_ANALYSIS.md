# Paired OP transition analysis

This analysis joins the saved traces by `record_id`. All 304 records were
present for one-pass, fixed-depth-2 and external-OP policies. Correctness is
the evaluator's post-generation `verified` field; the traces do not expose
the literal answer strings.

## One pass versus external OP

| One-pass state → OP state | Count |
|---|---:|
| Correct → Correct | 52 |
| Correct → Wrong | 0 |
| Wrong → Correct | 5 |
| Wrong → Wrong | 247 |

The external controller's net gain is therefore five genuine repairs with no
observed regressions in this sample. The repairs are three BBH records and two
ETHICS records:

- `bbh:boolean_expressions:00000195`
- `bbh:boolean_expressions:00000224`
- `bbh:boolean_expressions:00000230`
- `ethics:commonsense/cm_test:00000016`
- `ethics:commonsense/cm_test:00000032`

All five repairs used two passes and the same external action pattern:
`answer → critique → counterexample → evidence_search → revise → stop`.
The average generated-token count rose from 9.47 for one pass to 52.40 for
external OP. This indicates occasional answer repair, not broad transfer.

Domain transition counts:

| Domain | Correct → Correct | Correct → Wrong | Wrong → Correct | Wrong → Wrong |
|---|---:|---:|---:|---:|
| BBH | 17 | 0 | 3 | 44 |
| Big-Bench Lite | 0 | 0 | 0 | 64 |
| ETHICS | 33 | 0 | 2 | 29 |
| MATH | 1 | 0 | 0 | 63 |
| SciBench | 1 | 0 | 0 | 47 |

## One pass versus forced fixed-depth 2

| One-pass state → fixed-depth-2 state | Count |
|---|---:|
| Correct → Correct | 7 |
| Correct → Wrong | 45 |
| Wrong → Correct | 2 |
| Wrong → Wrong | 250 |

Forced second-pass reasoning damaged 45 previously correct records and
repaired only two. The largest damage was in BBH (17 correct-to-wrong) and
ETHICS (26 correct-to-wrong). This is answer drift, not merely failure to
improve.

## Interpretation

The paired analysis strengthens the diagnosis of the aggregate results:

- External OP occasionally repairs an answer and did not regress a correct
  answer in this single 304-record evaluation.
- The five repairs are too few to establish general reasoning improvement;
  this is a single checkpoint, one seed and a capped pilot set.
- Fixed-depth recursion is actively unstable for the current model. The model
  does not reliably preserve a correct answer through a forced second pass.
- The OP controller's information-gain and self-quality terms were unavailable
  or uncalibrated, so the action sequence is externally bounded rather than a
  learned internal recursive policy.

An exact two-sided McNemar calculation on the five OP discordant records would
give `p = 0.0625`; this is reported only as a descriptive check, not as a
confirmatory claim because the evaluation was not preregistered for this test.

## Recommended next engineering action

Before more foundation training, instrument the evaluator to retain each
pass's answer text or canonical answer hash, intermediate-state digest and
verification reason. Then classify fixed-depth failures as answer drift,
formatting failure, verifier mismatch or context truncation. That diagnostic
is needed before changing recursion depth or architecture; the current traces
support the behavioral conclusion but cannot identify the internal cause.
