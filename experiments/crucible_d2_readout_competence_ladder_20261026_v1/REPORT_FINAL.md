# Crucible D2 Readout Competence Ladder v1

CPU diagnostic separating readout competence, frozen final-state usability and end-to-end late-authority selection.

## Results

The competence gate is 95% training pointer accuracy. Paired flips and paired correctness are reported for two-candidate stages.

| Stage | Interface | Seed | Train pointer | Validation pointer | Validation flips | Validation paired correct |
|---|---|---:|---:|---:|---:|---:|
| S0_SYNTHETIC_HEAD | synthetic | 1337 | 80/80 | 40/40 | n/a | n/a |
| S0_SYNTHETIC_HEAD | synthetic | 42 | 80/80 | 40/40 | n/a | n/a |
| S1_SINGLE_CANDIDATE | canonical | 1337 | 80/80 | 20/20 | n/a | n/a |
| S1_SINGLE_CANDIDATE | canonical | 42 | 80/80 | 20/20 | n/a | n/a |
| S1_SINGLE_CANDIDATE | factorized | 1337 | 80/80 | 20/20 | n/a | n/a |
| S1_SINGLE_CANDIDATE | factorized | 42 | 80/80 | 20/20 | n/a | n/a |
| S2_DIRECT_TARGET | canonical | 1337 | 120/160 | 29/40 | 9/20 | 9/20 |
| S2_DIRECT_TARGET | canonical | 42 | 159/160 | 39/40 | 19/20 | 19/20 |
| S2_DIRECT_TARGET | factorized | 1337 | 85/160 | 21/40 | 1/20 | 1/20 |
| S2_DIRECT_TARGET | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 |
| S3_AUTHORITY_CONDITIONED | canonical | 1337 | 120/160 | 31/40 | 11/20 | 11/20 |
| S3_AUTHORITY_CONDITIONED | canonical | 42 | 160/160 | 39/40 | 19/20 | 19/20 |
| S3_AUTHORITY_CONDITIONED | factorized | 1337 | 91/160 | 26/40 | 6/20 | 6/20 |
| S3_AUTHORITY_CONDITIONED | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | canonical | 1337 | 160/160 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | canonical | 42 | 160/160 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | factorized | 1337 | 160/160 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | factorized | 42 | 160/160 | 40/40 | 20/20 | 20/20 |

## Unseen-layout transfer

S3 and S4 were evaluated on the preregistered validation-pair layout with the candidate block moved after the operands and neutral late padding inserted. This is exploratory transfer beyond the standard validation layout.

| Stage | Interface | Seed | Pointer | Flips | Paired correct |
|---|---|---:|---:|---:|---:|
| S3_AUTHORITY_CONDITIONED | canonical | 1337 | 20/40 | 0/20 | 0/20 |
| S3_AUTHORITY_CONDITIONED | canonical | 42 | 28/40 | 8/20 | 8/20 |
| S3_AUTHORITY_CONDITIONED | factorized | 1337 | 20/40 | 0/20 | 0/20 |
| S3_AUTHORITY_CONDITIONED | factorized | 42 | 20/40 | 0/20 | 0/20 |
| S4_END_TO_END_AUTHORITY | canonical | 1337 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | canonical | 42 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | factorized | 1337 | 40/40 | 20/20 | 20/20 |
| S4_END_TO_END_AUTHORITY | factorized | 42 | 40/40 | 20/20 | 20/20 |

## Interpretation

S0 tests the readout and target convention on an explicitly encoded synthetic feature. S1 tests the same linear readout on final decision states from frozen accepted D1 backbones with one candidate. S2 replaces the synthetic feature with a late, explicit candidate-slot cue. S3 replaces that direct cue with the actual late authority cue. S4 allows the dense backbone and final decision-pointer head to adapt jointly under decision-only loss.

A stage that fails its training competence gate is not interpreted as a held-out generalization failure. The S3 frozen-state result therefore distinguishes whether a simple readout can use the accepted D1 final state, while S4 tests whether end-to-end adaptation can make the late-authority pathway learnable. These results do not establish general instruction hierarchy or autonomous reasoning.

## Frozen control

Accepted D1 backbone checkpoints were verified against the prior experiment manifest. No sealed test data were accessed and no accepted checkpoint was modified.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_D2_READOUT_COMPETENCE_LADDER_PROTOCOL_20261026.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Data audit](DATA_AUDIT.json)
- [Protocol manifest](PROTOCOL_MANIFEST.json)
- Per-arm readouts and checkpoints are under `trained_arms/`
