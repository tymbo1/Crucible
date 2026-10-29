# Crucible 120M continuation to 25M tokens

Classification: `substantive_local_foundation_continuation`

This continuation resumed the existing step-5,000 checkpoint and ran the
unchanged 120M Crucible foundation configuration locally on the RTX 3060 Ti.
It is an exploratory language-foundation run, not a reasoning-competence or
OP result.

## Run record

| Item | Result |
|---|---:|
| Model parameters | 120,842,496 |
| Resumed checkpoint | step 5,000 |
| Final checkpoint | step 15,000 |
| Cumulative training tokens | 25,270,547 |
| Additional tokens in continuation | 16,844,680 |
| Continuation wall time | 4,917.71 s (1.37 h) |
| Continuation throughput | 3,425 tokens/s |
| Combined 0-to-15,000 wall time | approximately 7,689.85 s (2.14 h) |
| Combined throughput | approximately 3,286 tokens/s |
| Peak allocated VRAM | 2.93 GB |
| Peak reserved VRAM | 3.33 GB |
| Peak resident CPU memory | 2.82 GB |
| Non-finite gradients | none reported |

The continuation used the expanded external FineWeb-Edu pack containing
100,001,142 train tokens and 10,003,500 validation tokens. Its manifest hash
is recorded in `results_gpu/FOUNDATION_TRAINING_REPORT.json`; raw data remains
outside Git.

## Validation checkpoints

| Step | Cumulative tokens | Language loss | Reasoning loss |
|---:|---:|---:|---:|
| 5,000 | 8,425,867 | 1.5398 | 2.5080 |
| 6,000 | 10,109,250 | 1.5379 | 2.7354 |
| 7,000 | 11,792,671 | 1.5195 | 2.9004 |
| 8,000 | 13,478,089 | 1.4763 | 2.9889 |
| 9,000 | 15,161,297 | 1.5231 | 3.4361 |
| 10,000 | 16,850,923 | 1.4337 | 3.1417 |
| 11,000 | 18,535,117 | 1.4416 | 3.1880 |
| 12,000 | 20,218,500 | 1.4390 | 3.1903 |
| 13,000 | 21,901,921 | 1.4483 | 3.3066 |
| 14,000 | 23,587,339 | 1.4267 | 3.3160 |
| 15,000 | 25,270,547 | 1.4214 | 3.1859 |

The language validation loss is lower at step 15,000 than at step 5,000,
while the reasoning proxy remains noisy and does not establish reasoning
competence. The trainer did not run OP evaluation, cumulative-learning tests,
or sealed reasoning tests.

## Final generation probe

The fixed greedy probes at step 15,000 were still immature:

| Prompt | Completion |
|---|---|
| `The goal of science is` | `the values of 320` |
| `In mathematics, a prime number is` | `14'` |
| `A good explanation should` | `ro numbers` |
| `Once upon a time` | empty |

These samples show that the model is not yet a useful conversational model;
they are retained as a fixed diagnostic, not as a capability claim.

## Artifacts

Resumable checkpoints are stored outside Git under
`/home/timbushnell/crucible_120m_checkpoints_20261029/`, including the final
`crucible_120m_foundation_step_015000.pt`. Sanitized metrics, validation
curves and hashes are committed. The prior 5,000-step result remains the
baseline; this continuation does not alter its interpretation.
