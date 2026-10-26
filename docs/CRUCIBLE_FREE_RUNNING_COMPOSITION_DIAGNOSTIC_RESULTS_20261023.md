# Crucible free-running composition diagnostic

Date: 2026-10-23

## Executive result

The intermediate-supervision checkpoints do not yet execute the two-step procedure reliably without a supplied intermediate state.

At the final 2,000-update checkpoint:

| Metric | Seed 42 | Seed 1337 | Mean |
|---|---:|---:|---:|
| Teacher-forced final accuracy | 35.94% | 40.63% | 38.28% |
| Free-running intermediate accuracy | 76.56% | 87.50% | 82.03% |
| Free-running delimiter accuracy | 100.00% | 100.00% | 100.00% |
| Free-running final accuracy | 31.25% | 39.06% | 35.16% |
| Free-running end-to-end exact accuracy | 31.25% | 39.06% | 35.16% |
| Final accuracy given a correct intermediate | 40.82% | 44.64% | 42.86% |
| Counterfactual final accuracy | 57.81% | 57.81% | 57.81% |
| Counterfactual no-change rate | 48.44% | 32.81% | 40.63% |

The model has learned much of the first step and the output format. The remaining failure is concentrated in applying the second operation to the intermediate state.

## Protocol

- No training was performed.
- Evaluated the frozen `intermediate_and_final` checkpoints from the prior experiment.
- Seeds: 42 and 1337.
- Checkpoints: 0, 500, 1,000, 1,500 and 2,000 updates.
- Same 64 disjoint validation records used by the training experiment.
- Greedy CPU decoding.
- No held-out or test evaluation; no GPU or cloud execution.

For the counterfactual condition, the supplied intermediate was changed from `m` to `(m + 1) mod 10`. The expected final answer was independently recomputed by applying the original second operation to the changed intermediate.

## Checkpoint trajectory

| Seed | Updates | Teacher-forced final | Free intermediate | Free final | Free end-to-end | Counterfactual final |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 500 | 18.75% | 10.94% | 15.63% | 3.13% | 15.63% |
| 42 | 1,000 | 20.31% | 37.50% | 7.81% | 6.25% | 37.50% |
| 42 | 1,500 | 29.69% | 56.25% | 20.31% | 18.75% | 48.44% |
| 42 | 2,000 | 35.94% | 76.56% | 31.25% | 31.25% | 57.81% |
| 1337 | 500 | 15.63% | 51.56% | 14.06% | 6.25% | 20.31% |
| 1337 | 1,000 | 40.63% | 89.06% | 39.06% | 37.50% | 65.63% |
| 1337 | 1,500 | 37.50% | 92.19% | 37.50% | 35.94% | 60.94% |
| 1337 | 2,000 | 40.63% | 87.50% | 39.06% | 39.06% | 57.81% |

At the final checkpoint, all 64 examples per seed received the correct comma delimiter in free-running decoding. Therefore, delimiter formatting is not the remaining bottleneck in this evaluation.

## Counterfactual response

Changing the supplied intermediate changed the model's final prediction in 51.56% of seed-42 cases and 67.19% of seed-1337 cases. This is evidence that the final prediction is not completely independent of the intermediate state.

However, the changed final was correct only 57.81% of the time for each seed. The model is therefore sensitive to the supplied state, but its learned transformation of that state is not yet reliable.

When the model's free-running intermediate was wrong at the final checkpoint, its final answer was also wrong in all observed cases: 15/15 for seed 42 and 8/8 for seed 1337. Among examples with a correct generated intermediate, final accuracy was only 40.82% and 44.64%. This localizes the dominant remaining error to the second-step operation rather than the delimiter or solely the first-step prediction.

## Interpretation

The diagnostic supports three conclusions:

1. Intermediate supervision produces a genuine improvement in learning the first operation and the structured output format.
2. The model can condition its final prediction on a supplied intermediate; counterfactual changes are not ignored entirely.
3. The model has not yet learned a reliable, free-running two-step computation. Explicit decomposition improved the interface, but did not solve second-step execution.

The counterfactual result is not a proof of causal understanding. The model may be using a learned token-to-output association rather than applying the declared operation. The task families are also still the familiar `add+add` and `add+mul` families, not unseen operation combinations.

## Integrity checks

The diagnostic output has matching hashes, and it only reads the prior checkpoints and validation manifests. The source training artifacts remain unchanged.

## Next gate

The next decisive experiment should train or evaluate on genuinely unseen operation combinations while retaining free-running evaluation. A useful follow-up would compare:

- teacher-forced intermediate supervision;
- free-running execution with the model's own intermediate fed back;
- a counterfactual consistency score against the declared second operation;
- held-out operation-pair combinations.

The present result is a positive representation and conditioning signal, but not yet evidence that Crucible has acquired a general compositional procedure.
