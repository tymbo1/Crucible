# Phase VII read-only checkpoint diagnostic

Verification pass: **yes**.

This diagnostic replayed the committed N0, N1 and N2 final checkpoints without optimizer calls, verified their predictions against the committed train/validation CSVs, and compared candidate-order, query-direction, error-stratum and representation summaries.

## Paired validation by model

| Model | Condition | Seed | Paired | Value accuracy | Candidate-order value preserved | Centroid probe validation |
|---|---|---:|---:|---:|---:|---:|
| N0_SHARED_SCORER | P12_PRISM | 1337 | 23/48 | 0.615 | 27/48 | 25/96 (0.260) |
| N0_SHARED_SCORER | P12_PRISM | 42 | 29/48 | 0.750 | 30/48 | 27/96 (0.281) |
| N0_SHARED_SCORER | R12_DIAMETER5 | 1337 | 16/48 | 0.531 | 18/48 | 22/96 (0.229) |
| N0_SHARED_SCORER | R12_DIAMETER5 | 42 | 18/48 | 0.646 | 20/48 | 38/96 (0.396) |
| N1_SHARED_LATENT | P12_PRISM | 1337 | 17/48 | 0.354 | 48/48 | 34/96 (0.354) |
| N1_SHARED_LATENT | P12_PRISM | 42 | 17/48 | 0.354 | 48/48 | 40/96 (0.417) |
| N1_SHARED_LATENT | R12_DIAMETER5 | 1337 | 11/48 | 0.229 | 48/48 | 32/96 (0.333) |
| N1_SHARED_LATENT | R12_DIAMETER5 | 42 | 14/48 | 0.292 | 48/48 | 26/96 (0.271) |
| N2_SHARED_LATENT_CYCLE | P12_PRISM | 1337 | 15/48 | 0.312 | 48/48 | 30/96 (0.312) |
| N2_SHARED_LATENT_CYCLE | P12_PRISM | 42 | 10/48 | 0.208 | 48/48 | 22/96 (0.229) |
| N2_SHARED_LATENT_CYCLE | R12_DIAMETER5 | 1337 | 11/48 | 0.229 | 48/48 | 18/96 (0.188) |
| N2_SHARED_LATENT_CYCLE | R12_DIAMETER5 | 42 | 17/48 | 0.354 | 48/48 | 38/96 (0.396) |

## Interpretation

The detailed machine-readable comparisons are in `CHECKPOINT_DIAGNOSTIC.json`; per-row replay outputs are in `CHECKPOINT_DIAGNOSTIC_PREDICTIONS.csv`.

The diagnostic does not train a probe, alter checkpoints, or establish a unique internal mechanism. A centroid probe is included only as a fixed descriptive test of whether final representations cluster by target value when centroids are computed from training representations.
