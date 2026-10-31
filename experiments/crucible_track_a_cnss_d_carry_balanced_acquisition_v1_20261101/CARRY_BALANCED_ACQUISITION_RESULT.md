# CNSS Skill D carry-balanced acquisition control - CPU result

Classification: `COMPLETED_AUTHORIZED_CARRY_BALANCED_ACQUISITION_CONTROL`.

The balanced arm used the existing 128-state candidate support, exact carry quotas and paired persisted step-zero states. No GPU was used; partner rerun remains unauthorized.

| Seed | Train | Validation | Test | Paired control test | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.0000 | 0.8750 | 0.8594 | 0.8594 | +0.0000 |
| 1337 | 1.0000 | 0.8438 | 0.8125 | 0.8594 | -0.0469 |
| 2025 | 1.0000 | 0.8594 | 0.8125 | 0.8594 | -0.0469 |

Mean balanced test accuracy: `0.828125`; mean paired-control test accuracy: `0.859375`; mean delta: `-0.031250`.

All checkpoints were finite, atomically saved, hash-validated, optimizer-validated and replay-checked in a fresh process.

The 95% individual D gate remains unchanged and partner rerun remains locked.
