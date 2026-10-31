# D carry-balanced paired error-transition audit

Classification: `READ_ONLY_PAIRED_ERROR_TRANSITION_AUDIT`.

This audit compares the original rank-32/lambda-0.1 control with the carry-balanced arm on the identical 64-state D test partition. No training or checkpoint mutation occurred.

| Seed | Preserved correct | Correct→wrong | Wrong→correct | Preserved wrong | Control test | Balanced test | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 53 | 2 | 2 | 7 | 0.8594 | 0.8594 | +0.0000 |
| 1337 | 51 | 4 | 1 | 8 | 0.8594 | 0.8125 | -0.0469 |
| 2025 | 52 | 3 | 0 | 9 | 0.8594 | 0.8125 | -0.0469 |

Common regressions in seeds 1337 and 2025: `[]`.
Regressions common to all seeds: `[]`.

The exact row-level transition records, including predictions and targets, are retained only in the sealed local artifact referenced by D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT_SEALED_HASH.json.

This audit is diagnostic evidence, not a causal explanation of why the balanced schedule changed the outputs.
