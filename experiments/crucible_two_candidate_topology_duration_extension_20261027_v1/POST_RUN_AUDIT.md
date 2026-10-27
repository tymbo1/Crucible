# Post-run audit: topology training-duration extension

Verification pass: **yes**.

The audit verified the immutable source hashes, all 2,000 trace steps, every recorded checkpoint, exact reproduction of the committed 500-step train/validation metrics, CSV metric recomputation, parameter count and final training-gate outcomes.

| Cell | 500-step reproduced | CSV metrics reproduced | Gate | First recorded gate |
|---|---|---|---|---:|
| P12_PRISM_SEED42 | yes | yes | PASS | 500 |
| P12_PRISM_SEED1337 | yes | yes | PASS | 750 |
| R12_DIAMETER5_SEED42 | yes | yes | PASS | 750 |
| R12_DIAMETER5_SEED1337 | yes | yes | PASS | 750 |

The duration extension is therefore valid as an acquisition/competence diagnostic. It does not, by itself, adjudicate whether graph topology, path length, effective resistance or a learned multi-hop mechanism caused the original transfer differences.
