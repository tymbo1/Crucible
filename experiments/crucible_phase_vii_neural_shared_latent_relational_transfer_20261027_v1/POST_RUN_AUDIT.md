# Post-run audit: Phase VII neural shared-latent relational transfer

Verification pass: **yes**.

The audit independently verified the eight-arm matrix, frozen source hashes, data-leakage preflight, N0 reference metrics, 2,000 trace steps per arm, checkpoint metrics, CSV metric recomputation, parameter counts, checkpoint hashes and the matched paired-transfer comparison.

| Cell | Trace | Metrics | Checkpoint | Gate | Model paired | N0 paired | Δ |
|---|---:|---|---|---|---:|---:|---:|
| N1_SHARED_LATENT_P12_PRISM_SEED42 | 2000/2000 | yes | yes | PASS | 0.354 | 0.604 | -0.250 |
| N1_SHARED_LATENT_P12_PRISM_SEED1337 | 2000/2000 | yes | yes | PASS | 0.354 | 0.479 | -0.125 |
| N1_SHARED_LATENT_R12_DIAMETER5_SEED42 | 2000/2000 | yes | yes | PASS | 0.292 | 0.375 | -0.083 |
| N1_SHARED_LATENT_R12_DIAMETER5_SEED1337 | 2000/2000 | yes | yes | PASS | 0.229 | 0.333 | -0.104 |
| N2_SHARED_LATENT_CYCLE_P12_PRISM_SEED42 | 2000/2000 | yes | yes | PASS | 0.208 | 0.604 | -0.396 |
| N2_SHARED_LATENT_CYCLE_P12_PRISM_SEED1337 | 2000/2000 | yes | yes | PASS | 0.312 | 0.479 | -0.167 |
| N2_SHARED_LATENT_CYCLE_R12_DIAMETER5_SEED42 | 2000/2000 | yes | yes | PASS | 0.354 | 0.375 | -0.021 |
| N2_SHARED_LATENT_CYCLE_R12_DIAMETER5_SEED1337 | 2000/2000 | yes | yes | PASS | 0.229 | 0.333 | -0.104 |

All eight trained arms passed the 95% acquisition gate. The matched N0 comparison is a negative transfer result for this preregistered implementation: no N1 or N2 arm exceeded the frozen dense reference on paired validation accuracy.

This does not establish that shared-latent relational models or cycle-consistency objectives cannot help under other representations, budgets or tasks. It establishes that these tested Phase VII configurations did not improve transfer here.
