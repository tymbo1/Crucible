# C1j CPU memory-applicability stress test

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Frozen scale

- Training candidates: `20000`.
- Validation candidates: `5000`.
- Untouched held-out test tasks: `10000`.
- Learner updates per seed: `3000`.
- Held-out families: `interval, multiset, tree, matrix`.
- Virtual memory sizes: `10, 100, 1000, 10000`.

## Aggregate test results

| Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | Reuse coverage | Abstention precision | False abstentions | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 0.8761 | 1.0000 | 1.145 | 0.0000 | 0.8935 | 1.0000 | 0 | 0.0000 |
| `hybrid` | 0.8737 | 0.9973 | 1.002 | 0.0104 | 0.9029 | 1.0000 | 0 | 0.0172 |
| `learned` | 0.8700 | 0.9930 | 0.994 | 0.0259 | 0.9032 | 1.0000 | 0 | 0.0172 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| `persistent` | 0.5815 | 0.5383 | 0.000 | 0.4185 | 1.0000 | 0.0000 | 0 | 0.0000 |
| `stateless` | 0.8761 | 1.0000 | 1.210 | 0.0000 | 0.0000 | 1.0000 | 0 | 0.0000 |

## Held-out-family results

| Family | Policy | Accuracy | Mean cost | Harmful transfer | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| `interval` | `deterministic` | 0.8798 | 1.118 | 0.0000 | 0.0000 |
| `interval` | `hybrid` | 0.8798 | 0.973 | 0.0000 | 0.0163 |
| `interval` | `learned` | 0.8760 | 0.966 | 0.0144 | 0.0163 |
| `interval` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `interval` | `persistent` | 0.6045 | 0.000 | 0.3955 | 0.0000 |
| `interval` | `stateless` | 0.8798 | 1.205 | 0.0000 | 0.0000 |
| `multiset` | `deterministic` | 0.8734 | 1.137 | 0.0000 | 0.0000 |
| `multiset` | `hybrid` | 0.8734 | 0.997 | 0.0000 | 0.0153 |
| `multiset` | `learned` | 0.8705 | 0.990 | 0.0121 | 0.0153 |
| `multiset` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `multiset` | `persistent` | 0.5799 | 0.000 | 0.4201 | 0.0000 |
| `multiset` | `stateless` | 0.8734 | 1.201 | 0.0000 | 0.0000 |
| `tree` | `deterministic` | 0.8726 | 1.147 | 0.0000 | 0.0000 |
| `tree` | `hybrid` | 0.8633 | 0.993 | 0.0401 | 0.0229 |
| `tree` | `learned` | 0.8595 | 0.986 | 0.0552 | 0.0229 |
| `tree` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `tree` | `persistent` | 0.5724 | 0.000 | 0.4276 | 0.0000 |
| `tree` | `stateless` | 0.8726 | 1.227 | 0.0000 | 0.0000 |
| `matrix` | `deterministic` | 0.8789 | 1.177 | 0.0000 | 0.0000 |
| `matrix` | `hybrid` | 0.8789 | 1.043 | 0.0000 | 0.0142 |
| `matrix` | `learned` | 0.8746 | 1.034 | 0.0211 | 0.0142 |
| `matrix` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `matrix` | `persistent` | 0.5705 | 0.000 | 0.4295 | 0.0000 |
| `matrix` | `stateless` | 0.8789 | 1.206 | 0.0000 | 0.0000 |

## Memory-size results

| Memory size | Policy | Accuracy | Mean cost | Harmful transfer | Reuse coverage |
| ---: | --- | ---: | ---: | ---: | ---: |
| `10` | `deterministic` | 0.8756 | 1.120 | 0.0000 | 0.9875 |
| `10` | `hybrid` | 0.8752 | 0.972 | 0.0016 | 0.9890 |
| `10` | `learned` | 0.8729 | 0.966 | 0.0104 | 0.9901 |
| `10` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `10` | `persistent` | 0.6157 | 0.000 | 0.3843 | 1.0000 |
| `10` | `stateless` | 0.8756 | 1.207 | 0.0000 | 0.0000 |
| `100` | `deterministic` | 0.8656 | 1.107 | 0.0000 | 0.9566 |
| `100` | `hybrid` | 0.8648 | 0.966 | 0.0034 | 0.9598 |
| `100` | `learned` | 0.8624 | 0.961 | 0.0132 | 0.9598 |
| `100` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `100` | `persistent` | 0.6182 | 0.000 | 0.3818 | 1.0000 |
| `100` | `stateless` | 0.8656 | 1.194 | 0.0000 | 0.0000 |
| `1000` | `deterministic` | 0.8892 | 1.170 | 0.0000 | 0.8823 |
| `1000` | `hybrid` | 0.8868 | 1.029 | 0.0106 | 0.8917 |
| `1000` | `learned` | 0.8810 | 1.019 | 0.0352 | 0.8917 |
| `1000` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `1000` | `persistent` | 0.5718 | 0.000 | 0.4282 | 1.0000 |
| `1000` | `stateless` | 0.8892 | 1.230 | 0.0000 | 0.0000 |
| `10000` | `deterministic` | 0.8736 | 1.182 | 0.0000 | 0.7534 |
| `10000` | `hybrid` | 0.8676 | 1.039 | 0.0294 | 0.7763 |
| `10000` | `learned` | 0.8634 | 1.031 | 0.0488 | 0.7763 |
| `10000` | `oracle` | 1.0000 | 0.000 | 0.0000 | 0.0000 |
| `10000` | `persistent` | 0.5215 | 0.000 | 0.4785 | 1.0000 |
| `10000` | `stateless` | 0.8736 | 1.210 | 0.0000 | 0.0000 |

## Condition results

| Condition | Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | False abstentions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `applicable` | `deterministic` | 1.0000 | 1.0000 | 0.320 | 0.0000 | 0 |
| `applicable` | `hybrid` | 1.0000 | 1.0000 | 0.240 | 0.0000 | 0 |
| `applicable` | `learned` | 1.0000 | 1.0000 | 0.240 | 0.0000 | 0 |
| `applicable` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `applicable` | `persistent` | 0.8924 | 0.8924 | 0.000 | 0.1076 | 0 |
| `applicable` | `stateless` | 1.0000 | 1.0000 | 1.300 | 0.0000 | 0 |
| `adaptation` | `deterministic` | 1.0000 | 1.0000 | 0.771 | 0.0000 | 0 |
| `adaptation` | `hybrid` | 0.9819 | 0.9819 | 0.216 | 0.0198 | 0 |
| `adaptation` | `learned` | 0.9814 | 0.9814 | 0.216 | 0.0203 | 0 |
| `adaptation` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `adaptation` | `persistent` | 0.8946 | 0.8946 | 0.000 | 0.1054 | 0 |
| `adaptation` | `stateless` | 1.0000 | 1.0000 | 1.366 | 0.0000 | 0 |
| `invalidated` | `deterministic` | 1.0000 | 1.0000 | 1.480 | 0.0000 | 0 |
| `invalidated` | `hybrid` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `invalidated` | `learned` | 1.0000 | 1.0000 | 1.396 | 0.0000 | 0 |
| `invalidated` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `invalidated` | `persistent` | 0.0710 | 0.0710 | 0.000 | 0.9290 | 0 |
| `invalidated` | `stateless` | 1.0000 | 1.0000 | 1.300 | 0.0000 | 0 |
| `contradictory` | `deterministic` | 1.0000 | 1.0000 | 1.480 | 0.0000 | 0 |
| `contradictory` | `hybrid` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `contradictory` | `learned` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `contradictory` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `contradictory` | `persistent` | 0.0445 | 0.0445 | 0.000 | 0.9555 | 0 |
| `contradictory` | `stateless` | 1.0000 | 1.0000 | 1.300 | 0.0000 | 0 |
| `misleading_similarity` | `deterministic` | 1.0000 | 1.0000 | 1.480 | 0.0000 | 0 |
| `misleading_similarity` | `hybrid` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `misleading_similarity` | `learned` | 1.0000 | 1.0000 | 1.397 | 0.0000 | 0 |
| `misleading_similarity` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `misleading_similarity` | `persistent` | 0.0556 | 0.0556 | 0.000 | 0.9444 | 0 |
| `misleading_similarity` | `stateless` | 1.0000 | 1.0000 | 1.300 | 0.0000 | 0 |
| `noisy_evidence` | `deterministic` | 1.0000 | 1.0000 | 1.980 | 0.0000 | 0 |
| `noisy_evidence` | `hybrid` | 1.0000 | 1.0000 | 1.900 | 0.0000 | 0 |
| `noisy_evidence` | `learned` | 0.9709 | 0.9709 | 1.848 | 1.0000 | 0 |
| `noisy_evidence` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `noisy_evidence` | `persistent` | 0.8807 | 0.8807 | 0.000 | 0.1193 | 0 |
| `noisy_evidence` | `stateless` | 1.0000 | 1.0000 | 1.800 | 0.0000 | 0 |
| `provenance_corrupt` | `deterministic` | 1.0000 | 1.0000 | 1.480 | 0.0000 | 0 |
| `provenance_corrupt` | `hybrid` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `provenance_corrupt` | `learned` | 1.0000 | 1.0000 | 1.400 | 0.0000 | 0 |
| `provenance_corrupt` | `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0 |
| `provenance_corrupt` | `persistent` | 0.8941 | 0.8941 | 0.000 | 0.1059 | 0 |
| `provenance_corrupt` | `stateless` | 1.0000 | 1.0000 | 1.300 | 0.0000 | 0 |
| `unresolvable` | `deterministic` | 0.0000 | 0.0000 | 0.180 | 0.0000 | 0 |
| `unresolvable` | `hybrid` | 0.0000 | 0.0000 | 0.100 | 0.0000 | 0 |
| `unresolvable` | `learned` | 0.0000 | 0.0000 | 0.100 | 0.0000 | 0 |
| `unresolvable` | `oracle` | 1.0000 | 0.0000 | 0.000 | 0.0000 | 0 |
| `unresolvable` | `persistent` | 0.8870 | 0.0000 | 0.000 | 0.1130 | 0 |
| `unresolvable` | `stateless` | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0 |

## Preregistered gates

- Deterministic safety gate: `True`.
- Learned primary gate: `False`.
- Hybrid safety gate: `False`.
- Test separation: `True`.

A failed gate is retained as a valid result. These synthetic outcomes do not establish neural reasoning or CMS2 training value.

## Source hashes

- `experiments/crucible_c1j_cpu_stress_20261021/PROTOCOL.md`: `7b03656a44bf07c1b0fc98dd013967d9e8377432962ff43eb74b5ed6b67386a8`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
