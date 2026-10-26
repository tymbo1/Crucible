# Crucible composition exposure-control results

Date: 2026-10-23  
Status: CPU-only continuous exposure diagnostic complete; C2 remains blocked.

## Scope

This experiment tested whether the 128-example composition failure was mainly
an insufficient-exposure problem. The dense architecture, optimizer, data
order and objective were held fixed. Each seed used one fresh initialization
and one continuous 2,000-update trajectory, with measurements at 500, 1,000,
1,500 and 2,000 updates. These are continuation checkpoints, not independent
restarts.

The 128 training examples were balanced between `add+add` and `add+mul`. Each
seed used the same balanced 64-example validation set from the corrected
scaling protocol, disjoint from the complete training pool. Held-out and test
splits were not evaluated.

## Results

| Seed | Updates | Train exact | Validation exact | Forgotten since prior checkpoint |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 500 | 37.50% | 14.06% | 0 |
| 42 | 1,000 | 89.06% | 6.25% | 1 |
| 42 | 1,500 | 93.75% | 3.13% | 7 |
| 42 | 2,000 | 78.91% | 6.25% | 24 |
| 1337 | 500 | 36.72% | 14.06% | 0 |
| 1337 | 1,000 | 56.25% | 14.06% | 10 |
| 1337 | 1,500 | 86.72% | 4.69% | 6 |
| 1337 | 2,000 | 88.28% | 9.38% | 13 |

## Interpretation

Additional exposure improved training fit substantially at intermediate
checkpoints, especially for seed 42 at 1,000–1,500 updates. It did not produce
transfer to the disjoint validation set: validation remained in the low
single-digit to low-teen range throughout, far below the 70% learnability
criterion.

The later checkpoints also show destructive interference rather than simple
monotonic learning. Seed 42 fell from 93.75% train exact at 1,500 updates to
78.91% at 2,000, with 24 examples becoming incorrect relative to the previous
checkpoint. Seed 1337 also recorded checkpoint-to-checkpoint forgetting,
although its final training accuracy remained higher.

The final operation-pair breakdown reinforces the distinction. At 2,000
updates, seed 42 reached 75.00% train exact on `add+add` and 82.81% on
`add+mul`, but only 0.00% and 12.50% validation exact respectively. Seed 1337
reached 85.94% and 90.63% train exact, but only 15.63% and 3.13% validation
exact. The model is fitting examples, not reliably executing the reusable
operation.

Gradient norms remained nonzero at all measured continuation checkpoints, so
this is not a dead-gradient result. The run recorded losses, gradient norms,
prediction-digit histograms, target ranks and probabilities at every checkpoint.

## Decision

The exposure hypothesis is not supported as a sufficient explanation. More
updates can improve memorization, but they do not solve compositional transfer
and can introduce forgetting.

Do not extend this schedule further, increase model size, compare E4 or use GPU
capacity for this question. The next CPU gate should test the learning signal:

- retain the corrected disjoint validation protocol;
- compare final-answer-only supervision against explicitly supervised
  intermediate-state composition;
- match attempted updates and report the extra supervision clearly; and
- measure intermediate accuracy, final accuracy, operation-pair accuracy and
  forgetting separately.

Intermediate supervision would be a diagnostic intervention, not evidence of
unassisted compositional reasoning unless the final-answer-only control also
improves on independent validation.

## Integrity and artifacts

Both seeds completed the 2,000-update continuation without runtime failure.
Training/validation input overlap was zero for both seeds. Twenty per-seed
artifact hashes were independently recomputed with zero mismatches. Held-out
and test evaluation remained disabled.

- Accepted results: `experiments/crucible_composition_exposure_control_20261023/`
- Runner: `tools/run_crucible_composition_exposure_control_20261023.py`
- Corrected scaling baseline:
  `docs/CRUCIBLE_COMPOSITION_SCALING_LADDER_RESULTS_20261023.md`
