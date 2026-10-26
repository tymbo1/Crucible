# Carry-Sensitive Arithmetic Gate — Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_CARRY_SENSITIVE_ARITHMETIC_PROTOCOL_20261024.md`  
Run directory: `experiments/crucible_carry_sensitive_arithmetic_20261024/`

## Outcome

The symbolic addition oracle and all data audits passed, but neither neural interface reached reliable training fit within 1,000 updates. The run therefore does not isolate a carry-specific reasoning failure.

## Final exact accuracy

| Arm | Seed | Train exact | Validation exact |
|---|---:|---:|---:|
| Canonical numeric | 42 | 24.48% | 2.78% |
| Canonical numeric | 1337 | 27.08% | 8.33% |
| Factorized digits | 42 | 31.77% | 13.89% |
| Factorized digits | 1337 | 16.67% | 2.78% |

The factorized representation learned faster for seed 42, but the effect did not replicate in seed 1337 and neither arm approached a reliable training-fit threshold.

## Validation by carry regime

| Arm | Seed | No carry | Ones carry | Cascading carry |
|---|---:|---:|---:|---:|
| Canonical numeric | 42 | 0.00% | 4.17% | 4.17% |
| Canonical numeric | 1337 | 8.33% | 12.50% | 4.17% |
| Factorized digits | 42 | 8.33% | 16.67% | 16.67% |
| Factorized digits | 1337 | 4.17% | 4.17% | 0.00% |

## Controls and integrity

- Two interfaces × two fixed seeds = four CPU arms.
- 192 training and 72 validation pairs per arm.
- All three requested carry regimes were present in both splits.
- Every legal input digit occurred in training: tens `1..9`, ones `0..9`.
- Zero train/validation pair overlap.
- Symbolic oracle: 100% on every split and arm.
- No original sealed test set was opened.
- GPU use was disabled.
- `git diff --check` passed.

Pairs with a tens carry but no ones carry were excluded as a separate fourth regime; they were not silently assigned to another category.

## Interpretation

The addition task is materially harder than the modulo-ten successor task. At the current update budget, the model has not fit the mixed two-digit task well enough for validation scores to distinguish no-carry, ones-carry and cascading-carry competence. The factorized prompt may provide a useful optimization signal, but the seed disagreement and low training fit prevent a strong representation claim.

This is not evidence that carry reasoning is impossible, nor that the architecture needs BDH, spiking neurons or Hebbian plasticity. It is a bounded learnability result: ordinary two-digit addition with variable-length outputs and mixed carry regimes is not learned reliably by this control under the present objective and budget.

## Decision

Keep the current architecture frozen. The next arithmetic gate should first establish a positive control for two-digit addition without carry and with fixed two-digit outputs, then add ones carry, then cascading carry. Each stage should pass a training-fit threshold before its held-out transfer is interpreted.
