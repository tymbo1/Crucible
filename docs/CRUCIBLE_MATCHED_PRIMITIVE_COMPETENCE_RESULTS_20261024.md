# Crucible Matched Primitive-Competence Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_MATCHED_PRIMITIVE_COMPETENCE_PROTOCOL_20261024.md`  
Execution: CPU-only dense control

## Result

The tiny-set overfit control passed for both seeds, but the preregistered primitive-transfer gate failed for both seeds.

| Seed | Arm | Train exact | Interpolation exact | Generalization exact | Probe exact | Gate |
|---:|---|---:|---:|---:|---:|---|
| 42 | Tiny, 8 contexts | 100.00% | — | — | 0.00% | overfit passed |
| 42 | Full, 72 contexts | 100.00% | 10.71% (3/28) | 35.00% (7/20) | — | interpolation failed |
| 1337 | Tiny, 8 contexts | 100.00% | — | — | 0.00% | overfit passed |
| 1337 | Full, 72 contexts | 100.00% | 10.71% (3/28) | 10.00% (2/20) | — | interpolation failed |

The interpolation threshold was frozen at 90% before training. The uniform one-digit baseline is 10%, so the two full arms are essentially at chance on interpolation despite fitting the training contexts perfectly.

## Per-operation breakdown

Final full-arm exact accuracy:

| Seed | Split | add | mul | sub |
|---:|---|---:|---:|---:|
| 42 | Interpolation | 0/12 | 3/12 | 0/4 |
| 42 | Generalization | — | 6/10 | 1/10 |
| 1337 | Interpolation | 0/12 | 3/12 | 0/4 |
| 1337 | Generalization | — | 2/10 | 0/10 |

All three operations reached 100% training accuracy in both full arms. The generalization holdout had no preregistered threshold and is small; its seed-dependent 35% versus 10% result is diagnostic only, not evidence of reliable transfer.

## What the gate establishes

This is a substantially cleaner primitive test than the earlier B-only arm:

- The input explicitly contains the supplied intermediate, operation and operand.
- The prompt uses the position-gate prefix and `step2=operation(operand)` form.
- All operations, supplied intermediate values and operand values appear in training.
- Interpolation examples use unseen complete contexts while retaining familiar individual feature values.
- Generalization withholds the operation–operand pairs `mul(4)` and `sub(1)` while retaining their individual factors in training.
- Independent oracle and duplicate checks passed before training.

The reproducible pattern is therefore:

> The dense control can memorize one-step arithmetic examples under the matched interface, but it does not learn a reusable primitive transformation under this data, objective and 2,000-update budget.

This moves the immediate bottleneck below two-step composition. It is not yet evidence that the broader Crucible architecture is incapable of arithmetic, nor that more parameters would solve it. It is evidence that adding recursive reasoning or changing operation order would be premature.

## Training dynamics

Seed 42 reached 79.17% training exact at step 500 and 100% by step 1,000. Interpolation was 7.14% at step 500 and 14.29% at step 1,000 before falling to 10.71% at the final checkpoint.

Seed 1337 reached 51.39% training exact at step 500 and 97.22% by step 1,000. Interpolation was 3.57% at step 500 and 10.71% at steps 1,000, 1,500 and 2,000.

Additional fitting therefore did not produce the required transfer. This is not explained by simply stopping too early at the tested budget.

## Limitations

- Two seeds and 28 interpolation examples are sufficient to fail the preregistered 90% gate, but not to characterize all possible representations or training schedules.
- The interpolation split tests unseen full contexts, not every possible definition of arithmetic generalization.
- The experiment uses the dense control, not E4 or the 170M Crucible model.
- The result does not distinguish all possible causes among token-level representation, optimization dynamics, sampling density and inductive bias.

## Decision

C2, E4 comparison and GPU scaling remain blocked. Do not introduce recursive supervision or architectural changes based solely on the two-step failure.

The next justified CPU study is a representation/data-interface ladder: retain the same one-step semantics, add controlled repeated coverage or an explicitly structured numeric encoding, and test whether interpolation improves before returning to composition. Any such change must be separately versioned and preregistered; these results remain frozen.

## Artifacts

- [Frozen protocol](CRUCIBLE_MATCHED_PRIMITIVE_COMPETENCE_PROTOCOL_20261024.md)
- [Runner](../tools/run_crucible_matched_primitive_competence_20261024.py)
- [Run summary](../experiments/crucible_matched_primitive_competence_20261024/SUMMARY.json)
- [Split manifest](../experiments/crucible_matched_primitive_competence_20261024/SPLIT_MANIFEST.json)
- [Hash manifest](../experiments/crucible_matched_primitive_competence_20261024/HASHES.json)
