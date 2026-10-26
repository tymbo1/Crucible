# Crucible intermediate-state supervision results

Date: 2026-10-23

## Executive result

Explicit intermediate-state supervision substantially improved validation performance on the two-step composition task for both seeds.

At 2,000 updates, final-token validation accuracy was:

| Seed | Final-only control | Intermediate + final | Difference |
|---:|---:|---:|---:|
| 42 | 6.25% | 35.94% | +29.69 pp |
| 1337 | 9.38% | 40.63% | +31.25 pp |
| Mean | 7.81% | 38.28% | +30.47 pp |

The supervised intermediate token was also learned strongly: 76.56% and 87.50% validation accuracy at step 2,000 for seeds 42 and 1337 respectively.

This is positive evidence that explicit decomposition improves transfer within the tested task family. It is not yet evidence of end-to-end compositional reasoning, because the reported token metrics use teacher-forced prefixes and the validation set uses the same `add+add` and `add+mul` operation families as training.

## Protocol

- CPU-only dense control with 21,968,640 parameters.
- Two seeds: 42 and 1337.
- Two paired arms per seed:
  - `final_only`: original one-digit target.
  - `intermediate_and_final`: intermediate digit and final digit target, separated by a comma.
- 128 training records and 64 balanced validation records per arm.
- Validation inputs were disjoint from training inputs; the two operation-pair families were balanced in both splits.
- 2,000 continuous updates per arm; checkpoints at 0, 500, 1,000, 1,500 and 2,000.
- Same initialization, data order, architecture and optimizer within each seed pair.
- No held-out or test evaluation; no GPU, cloud execution or language-model training.

The experimental arm has four supervised target positions per example rather than two in the control. It therefore is not a compute-matched architectural comparison; the added process targets are the intervention being tested.

## Validation trajectory

Final-token accuracy, with the intermediate arm's final position evaluated under teacher forcing:

| Updates | Final-only, seed 42 | Intermediate, seed 42 | Final-only, seed 1337 | Intermediate, seed 1337 |
|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0.00% | 0.00% |
| 500 | 14.06% | 18.75% | 14.06% | 15.63% |
| 1,000 | 6.25% | 20.31% | 14.06% | 40.63% |
| 1,500 | 3.13% | 29.69% | 4.69% | 37.50% |
| 2,000 | 6.25% | 35.94% | 9.38% | 40.63% |

At step 2,000, validation intermediate-token accuracy was 76.56% for seed 42 and 87.50% for seed 1337. Validation accuracy requiring both intermediate and final target positions to be correct was 31.25% and 39.06% respectively.

The corresponding final training accuracies were 96.88% and 96.88% for the intermediate arm, versus 78.91% and 88.28% for the final-only control. This confirms improved fitting as well as improved validation transfer, although it does not by itself establish a learned reusable algorithm.

## Runtime and artifact checks

| Seed | Final-only runtime | Intermediate runtime | Increase |
|---:|---:|---:|---:|
| 42 | 612.8 s | 834.0 s | 36.1% |
| 1337 | 606.1 s | 831.2 s | 37.1% |

The intermediate arm's extra target positions and longer target sequence increase runtime, so the result should not be described as a matched-compute improvement.

All four arm hash manifests verified with zero mismatches. Each arm contains 128 training and 64 validation records, with zero input overlap between those splits.

## Interpretation and limitation

The result supports the hypothesis that supervising the intermediate state gives the model a more useful learning signal than final-answer-only supervision. The intermediate state is learned substantially better than the final composition, which localizes a remaining difficulty in applying the learned intermediate result to the second operation.

There is an important measurement limitation: the current `final_token_accuracy` for the experimental arm is measured at the final target position while the correct intermediate token is present in the teacher-forced prefix. The `intermediate_and_final_accuracy` metric is also teacher-forced. These metrics therefore measure conditional supervised composition, not free-running generation in which the model must feed its own intermediate prediction into the second step.

The experiment also does not test unseen operation pairs, the sealed held-out split, larger models, E4, recursive control or cognitive competence. C2 remains blocked.

## Next gate

Before changing the architecture, run a bounded follow-up with:

1. Free-running/autoregressive evaluation of the intermediate-plus-final format.
2. A final-answer-only control with matched target-token and compute accounting where practical.
3. Held-out operation combinations, not only new inputs from the trained `add+add` and `add+mul` families.
4. Separate intermediate, final-conditional and end-to-end exact metrics.

The current result is a positive learning-interface result, not yet proof that Crucible has acquired a general compositional procedure.
