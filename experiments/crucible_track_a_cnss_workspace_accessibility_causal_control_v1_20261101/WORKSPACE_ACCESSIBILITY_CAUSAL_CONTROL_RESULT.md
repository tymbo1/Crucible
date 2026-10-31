# CNSS workspace-accessibility causal control - CPU result

Classification: `COMPLETED_AUTHORIZED_WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL`.

The alternative signed-bit workspace was trained with the same rank-32 adapter, lambda, paired initial states, standard schedule, seeds and fixed partitions as the existing original-workspace control. No GPU was used and partner rerun remains locked.

| Seed | Alternative train | Alternative validation | Alternative test | Original test | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.0000 | 0.8438 | 0.8438 | 0.8594 | -0.0156 |
| 1337 | 1.0000 | 0.6875 | 0.6875 | 0.8594 | -0.1719 |
| 2025 | 1.0000 | 0.6250 | 0.7344 | 0.8594 | -0.1250 |

Mean alternative test accuracy: `0.755208`; mean original-workspace test accuracy: `0.859375`; mean delta: `-0.104167`.

Decisive workspace-support criterion passed: `False`. Partner rerun unlocked: `False`.

All checkpoints were finite, hash-validated, optimizer-validated and replay-checked in a fresh process.
