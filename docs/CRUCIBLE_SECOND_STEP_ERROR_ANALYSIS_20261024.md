# Crucible second-step error analysis

Date: 2026-10-24

## Scope

This is a read-only analysis of the frozen intermediate-supervision checkpoints and free-running traces from 2026-10-23. No model weights were changed and no retraining occurred.

The existing adapter supports a meaningful conditional second-step probe: the correct intermediate digit and comma are supplied, and the model's final-token prediction is measured. This is not an isolated new prompt format; it is the same trained composition interface under teacher forcing.

## Final-checkpoint decomposition

| Seed | Intermediate correct | Conditional second-step probe | Free final | Free final given correct intermediate | End-to-end |
|---:|---:|---:|---:|---:|---:|
| 42 | 76.56% | 35.94% | 31.25% | 40.82% | 31.25% |
| 1337 | 87.50% | 40.63% | 39.06% | 44.64% | 39.06% |

When the free-running intermediate was wrong, the final answer was also wrong in every observed final-checkpoint case: 15/15 for seed 42 and 8/8 for seed 1337. The delimiter was correct in all 128 final-checkpoint free-running cases, so formatting is not the bottleneck.

This localizes the dominant failure to applying the second operation, while also showing that first-step errors propagate rather than being corrected downstream.

## Operation-pair and second-operation breakdown

Final checkpoint, 32 validation examples per pair:

| Seed | Pair / second op | Intermediate accuracy | Conditional second-step probe | Free end-to-end | Free final given correct intermediate |
|---:|---|---:|---:|---:|---:|
| 42 | `add+add` / add | 81.25% | 28.13% | 21.88% | 26.92% |
| 42 | `add+mul` / mul | 71.88% | 43.75% | 40.63% | 56.52% |
| 1337 | `add+add` / add | 81.25% | 50.00% | 46.88% | 57.69% |
| 1337 | `add+mul` / mul | 93.75% | 31.25% | 31.25% | 33.33% |

The second operation is not uniformly harder by type: `add` is weaker for seed 42, while `mul` is weaker for seed 1337. This points to seed-, input- and representation-dependent interference rather than a simple operation ranking.

## Target-value variation

The analysis also records breakdowns by expected intermediate digit and expected final digit in `SUMMARY.json`. These strata vary substantially, but many contain only 2–13 examples per seed. For example, seed 42's free end-to-end accuracy by expected final digit ranges from 0% to 66.67%, while seed 1337 ranges from 16.67% to 60.00%. These are diagnostic patterns, not reliable digit-specific estimates.

The variation reinforces the need for balanced target-value coverage in the next training gate. It does not by itself establish a digit-specific mechanism.

## Interpretation

The current evidence supports:

1. The model often computes the first operation on unseen validation inputs.
2. The model can use a supplied intermediate to produce a final prediction, but only inconsistently.
3. Free-running errors are concentrated in second-step application after the intermediate is available.
4. A correct first step is necessary but not sufficient for a correct final answer.

The conditional probe should not be called autonomous second-step reasoning: it supplies the correct intermediate and uses teacher forcing. A genuine isolated second-step prompt would be out-of-distribution because the model was not trained on that prompt format; evaluating it would measure interface transfer as well as arithmetic ability.

## Recommended next gate

Do not increase model size yet. The next controlled training experiment should preserve the current intermediate-supervision format while adding balanced coverage of operation position:

- expose the same operations in both first and second positions;
- include held-out operation-pair combinations;
- retain free-running evaluation;
- report conditional second-step, free-running final and end-to-end exact accuracy separately;
- retain counterfactual consistency checks;
- balance intermediate and final target digits.

That experiment can distinguish a second-position/interference problem from a broader inability to learn the operation itself.

## Artifact integrity

The analysis output was generated from the existing frozen traces and validation manifests. Its hash manifest matched with zero mismatches. No GPU or cloud resources were used.
