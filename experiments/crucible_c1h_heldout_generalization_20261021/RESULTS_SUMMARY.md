# C1h held-out structural generalization

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate results

| Arm | Accuracy | Novel-structure accuracy | Reuse coverage | Adaptation success | Harmful transfer | Abstention precision | False abstentions | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 0.8571 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 1.114 |
| `oracle` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.000 |
| `persistent` | 0.5714 | 1.0000 | 0.9167 | 0.0000 | 0.4390 | 0.0000 | 0 | 0.031 |
| `safe_adaptive` | 0.8571 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.136 |
| `shuffled_memory` | 0.8571 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.136 |
| `stateless` | 0.8571 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 1.114 |

## Condition results

### novel_surface

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `safe_adaptive` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `shuffled_memory` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### novel_structure

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `shuffled_memory` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### adaptation

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.083 |
| `shuffled_memory` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.083 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### invalidated

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `shuffled_memory` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### contradictory

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `shuffled_memory` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### misleading_similarity

| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_disabled` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `safe_adaptive` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
| `shuffled_memory` | 1.0000 | 0.8333 | 0.0000 | 0.0000 | 0.217 |
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
| `memory_disabled` | 0 | 0 | 72 | 0 |
| `oracle` | 0 | 0 | 0 | 0 |
| `persistent` | 24 | 0 | 0 | 0 |
| `safe_adaptive` | 72 | 24 | 72 | 0 |
| `shuffled_memory` | 77 | 29 | 72 | 0 |
| `stateless` | 0 | 0 | 72 | 0 |

## Preregistered gates

- Primary safe-generalization gate: `True`.
- Novel-structure gate: `True`.
- Adaptation gate: `True`.
- Shuffled-memory harmful-transfer gate: `True`.
- Contradiction-preservation gate: `True`.
- False abstentions are reported separately; abstention alone is not treated as success.

## Interpretation

A passing result transfers the generic controller contract to these synthetic held-out families. It does not demonstrate learned applicability, neural reasoning or CMS2-assisted training.

## Source hashes

- `experiments/crucible_c1h_heldout_generalization_20261021/PROTOCOL.md`: `a3e82eb91e2f8c3d8e6d7419bfb5ca4eb1abc63d540d43706ade2060d7b5e2d9`
- `tools/run_crucible_c1f_persistent_reuse_20261021.py`: `8a8af7d4a12e4b9fd124331957ef688867915857f9136e86dc4d41d1dd3f2343`
- `tools/run_crucible_c1g_safe_reuse_20261021.py`: `d943888d7cb09c2c74d8b04685c70d754bacd996b5a46656fe36c399514e6de1`
- `tools/run_crucible_c1h_heldout_generalization_20261021.py`: `adc3f291a3fbaeca8eab62f7521a1b8ab0de6b32d5fb5c2331333f221e64a36d`
