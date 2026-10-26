# Crucible D2 Component Ablation v1

2x2 CPU ablation of candidate-slot versus operation-class readout and decision-only versus auxiliary multi-task loss.

## Results

The training competence gate is 95% selection accuracy. Paired flips and paired correctness are the authority counterfactual metrics.

| Cell | Interface | Seed | Train selection | Validation selection | Validation flips | Paired correct | Source |
|---|---|---:|---:|---:|---:|---:|---|
| C1_SLOT_DECISION_ONLY | canonical | 42 | 160/160 | 40/40 | 20/20 | 20/20 | reused_accepted_s4 |
| C1_SLOT_DECISION_ONLY | canonical | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | reused_accepted_s4 |
| C1_SLOT_DECISION_ONLY | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 | reused_accepted_s4 |
| C1_SLOT_DECISION_ONLY | factorized | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | reused_accepted_s4 |
| C2_SLOT_FULL_MULTITASK | canonical | 42 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C2_SLOT_FULL_MULTITASK | canonical | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C2_SLOT_FULL_MULTITASK | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C2_SLOT_FULL_MULTITASK | factorized | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C3_OPERATION_DECISION_ONLY | canonical | 42 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C3_OPERATION_DECISION_ONLY | canonical | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C3_OPERATION_DECISION_ONLY | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C3_OPERATION_DECISION_ONLY | factorized | 1337 | 160/160 | 40/40 | 20/20 | 20/20 | new_crossed_training |
| C4_OPERATION_FULL_MULTITASK | canonical | 42 | 79/160 | 20/40 | 0/20 | 0/20 | reused_accepted_d2 |
| C4_OPERATION_FULL_MULTITASK | canonical | 1337 | 80/160 | 20/40 | 0/20 | 0/20 | reused_accepted_d2 |
| C4_OPERATION_FULL_MULTITASK | factorized | 42 | 80/160 | 20/40 | 0/20 | 0/20 | reused_accepted_d2 |
| C4_OPERATION_FULL_MULTITASK | factorized | 1337 | 80/160 | 20/40 | 0/20 | 0/20 | reused_accepted_d2 |

## Interpretation

C1 is the accepted S4 candidate-slot/decision-only positive control. C4 is the
accepted original D2 operation-class/multi-task negative control. C2 and C3
are the newly trained crossed cells.

All eight C1/C2 arms and all four C3 arms passed the training competence gate
and achieved 40/40 standard validation selection, 20/20 counterfactual flips
and 20/20 two-way paired correctness. The four C4 arms remained below the
training gate at 79--80/160 selection accuracy and therefore their 20/40
validation selection and 0/20 paired results are reported but are not clean
held-out generalization evidence.

Within this preregistered comparison, the successful result does not require
the candidate-slot readout: the newly trained operation-class/decision-only
C3 cell passed across both interfaces and both seeds. The auxiliary
multi-task objective also did not prevent success with the candidate-slot
readout: C2 passed across all four arms. The failure is confined to the
operation-class/full-multi-task configuration represented by the reused
historical D2 checkpoints.

That last comparison is not a fully matched causal ablation of the objective
alone. C4 was deliberately reused from the accepted original D2 experiment,
including its original training trajectory and schedule, while C2/C3 were
newly trained by this runner. The supported conclusion is therefore that the
original D2 combination or training trajectory is the failing configuration;
isolating objective from trajectory would require a separately preregistered
matched retraining of C4.

No architecture was changed, no sealed test data were accessed, and no result
below the competence gate is interpreted as independent transfer evidence.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_D2_COMPONENT_ABLATION_PROTOCOL_20261026.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Data audit](DATA_AUDIT.json)
- [Protocol manifest](PROTOCOL_MANIFEST.json)
- Per-cell checkpoints and traces are under `trained_arms/`
