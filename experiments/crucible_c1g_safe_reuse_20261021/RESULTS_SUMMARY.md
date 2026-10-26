# C1g evidence-grounded adaptive reuse

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate results

| Arm | Accuracy | Reuse coverage | Adaptation success | Harmful transfer | Abstention precision | False abstentions | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 1.040 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.000 |
| `persistent` | 0.4000 | 1.0000 | 0.0000 | 0.6000 | 0.0000 | 0 | 0.000 |
| `safe_adaptive` | 0.8000 | 0.9375 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.081 |
| `shuffled_memory` | 0.8000 | 0.9375 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.081 |
| `stateless` | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 1.040 |

## Condition results

### applicable

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `shuffled_memory` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### misleading_similarity

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `shuffled_memory` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### changed_environment

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.8750 | 0.0000 | 0.0000 | 0.163 |
| `shuffled_memory` | 1.0000 | 0.8750 | 0.0000 | 0.0000 | 0.163 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### contradictory

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `shuffled_memory` | 1.0000 | 0.9375 | 0.0000 | 0.0000 | 0.081 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### unresolvable

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `shuffled_memory` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `stateless` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |

## Memory transactions

| Arm | Validated writes | Contradiction records | Provisional records | Failed records |
| --- | ---: | ---: | ---: | ---: |
| `memory_disabled` | 0 | 0 | 144 | 0 |
| `oracle` | 0 | 0 | 0 | 0 |
| `persistent` | 36 | 0 | 0 | 0 |
| `safe_adaptive` | 63 | 36 | 144 | 0 |
| `shuffled_memory` | 75 | 39 | 144 | 0 |
| `stateless` | 0 | 0 | 144 | 0 |

## Preregistered gates

- Primary safe-reuse gate: `True`.
- Shuffled-memory harmful-transfer gate: `True`.
- Contradiction preservation: `True`.
- False abstentions are reported separately; abstention alone is not treated as success.

## Interpretation

The persistent arm is the primary C1f baseline. Safe adaptive reuse is only accepted if it improves applicability decisions without increasing unsafe transfer. A failed gate is preserved as an experimental result.

## Source hashes

- `experiments/crucible_c1g_safe_reuse_20261021/PROTOCOL.md`: `20984f323339a47f01ed840051be6336daec9734308f2c7015ba5c231841e95e`
- `tools/run_crucible_c1f_persistent_reuse_20261021.py`: `8a8af7d4a12e4b9fd124331957ef688867915857f9136e86dc4d41d1dd3f2343`
- `tools/run_crucible_c1g_safe_reuse_20261021.py`: `d943888d7cb09c2c74d8b04685c70d754bacd996b5a46656fe36c399514e6de1`
