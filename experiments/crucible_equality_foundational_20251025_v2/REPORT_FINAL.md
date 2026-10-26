# Crucible Equality Foundational Learning — Final Analysis

CPU-only, 12 arms, same diagonal addition split and 500 addition updates per arm. No sealed test access.

Addition fit gate: 11/12 arms passed. Verification: PASS.

## Addition results

| Arm | Validation accuracy by seed/interface | Special diagonal result |
|---|---|---:|
| A0_original | 60.0%, 60.0%, 60.0%, 46.7% | 0/8 |
| A1_equality_pretrained | 6.7%, 40.0%, 53.3%, 40.0% | 0/8 |
| A2_explicit_equality | 53.3%, 46.7%, 60.0%, 60.0% | 0/8 |

## Equality pretraining

A1 received 500 equality-task updates before addition fine-tuning. Equality pretraining results and elapsed time are recorded in `ANALYSIS_FINAL.json`; its addition optimizer was reset before the 500 addition updates.

A1 did not produce a reliable transfer benefit. One A1 arm failed the addition training-competence gate (75% training accuracy), while the other A1 arms fit but remained below the A0 diagonal baseline on validation.

A2 supplied equality explicitly as a diagnostic upper-bound feature. It also did not restore the held-out diagonal transfer in this configuration.

## Interpretation

These results do not show that equality is irrelevant. They show that a short equality pretraining phase and a binary prompt feature were insufficient to produce reliable transfer under this task and budget. The next foundational target should be variable binding and relation-conditioned execution, not persistent memory or recursion yet.

## Artifacts

- [Final analysis](ANALYSIS_FINAL.json)
- [Prediction matrix](PREDICTIONS_WITH_FIT.csv)
- [Shared per-arm diagnostics](forensics/arm_outputs/)
- [Heatmaps and confusion matrices](forensics/plots/)
- [Protocol](../../docs/CRUCIBLE_EQUALITY_FOUNDATIONAL_PROTOCOL_20251025.md)
- [Preflight](PREFLIGHT.json)
- [Initial anchors](INITIAL_ANCHORS.json)
