# Staged Arithmetic Curriculum versus Direct Training — Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_STAGED_ARITHMETIC_CURRICULUM_PROTOCOL_20261024.md`  
Run directory: `experiments/crucible_staged_arithmetic_curriculum_20261024/`

## Outcome

The curriculum improved training fit relative to direct training for seed 42, but it did not improve held-out validation. Every preregistered stage-promotion gate failed.

## Final matched-budget comparison

Both arms received 3,000 updates per seed using the factorized digit representation.

| Arm | Seed | Final train exact | Final validation exact |
|---|---:|---:|---:|
| Curriculum | 42 | 87.50% | 6.94% |
| Direct | 42 | 62.50% | 6.94% |
| Curriculum | 1337 | 83.85% | 2.78% |
| Direct | 1337 | 93.23% | 6.94% |

Across seeds, curriculum training fit averaged 85.68% versus 77.86% for direct training, but validation averaged 4.86% versus 6.94%. This is not evidence of a curriculum transfer benefit.

## Promotion gates

| Seed | Stage | Train | Validation | Promotion |
|---:|---|---:|---:|---|
| 42 | No carry | 100.00% | 8.33% | Failed |
| 42 | No + ones carry | 94.53% | 2.08% | Failed |
| 42 | All three regimes | 87.50% | 6.94% | Failed |
| 1337 | No carry | 92.19% | 4.17% | Failed |
| 1337 | No + ones carry | 93.75% | 0.00% | Failed |
| 1337 | All three regimes | 83.85% | 2.78% | Failed |

The curriculum therefore never met the required 95% training and 90% validation thresholds. Later stages were run as diagnostic continuations, not as validated curriculum promotions.

## Controls and integrity

- Two fixed seeds; curriculum and direct arms matched at 3,000 updates.
- Stage data contained the intended no-carry, ones-carry and cascading-carry regimes.
- Zero oracle errors and zero train/validation pair overlap.
- No original sealed test set was opened.
- The prior successor checkpoint was preserved and hash-referenced unchanged: `823234e6226a18b30897e7383d10a9630046019585103c1845cdcf8170477ef7`.
- GPU use was disabled.
- `git diff --check` passed.

## Interpretation

Staging made the training distribution easier at early stages and produced a higher final training fit for seed 42, but the benefit did not transfer to held-out addition pairs. Direct training performed better on validation for seed 1337 and tied the curriculum for seed 42.

The current evidence supports an optimization/fit effect, not a demonstrated curriculum advantage. Because the first no-carry validation gate failed, the experiment does not establish that the model learned a reusable no-carry addition procedure before carry was introduced.

## Decision

Keep the architecture and curriculum claim frozen. The next useful gate is a smaller positive-control ladder: fixed no-carry addition with enough repeated exposure to reach the 95% training threshold, followed by held-out no-carry transfer. Only after that gate passes should ones-carry and cascading-carry be introduced. The successful successor checkpoint remains a representation-transfer control, not an initialization for arithmetic training.
