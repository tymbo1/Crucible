# C1i learned versus deterministic applicability

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate test results

| Policy | Accuracy | Novel-structure accuracy | Reuse coverage | Harmful transfer | Abstention precision | False abstentions | Brier | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 0.8890 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.0000 | 0.830 |
| `hybrid` | 0.8890 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.0006 | 0.919 |
| `learned` | 0.8890 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.0006 | 0.635 |
| `oracle` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.000 |
| `persistent` | 0.7250 | 1.0000 | 0.0000 | 0.2750 | 0.0000 | 0 | 0.0000 | 0.798 |
| `stateless` | 0.8890 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0.0000 | 1.156 |

## Condition results

### novel_surface

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `hybrid` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.100 |
| `learned` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.100 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### novel_structure

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `hybrid` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `learned` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.100 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### adaptation

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.500 |
| `hybrid` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.600 |
| `learned` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.100 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### invalidated

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `hybrid` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `learned` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### contradictory

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `hybrid` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `learned` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### misleading_similarity

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `hybrid` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `learned` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.400 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |
| `stateless` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.300 |

### unresolvable

| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `hybrid` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `learned` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| `oracle` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `persistent` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| `stateless` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.000 |

## Learner and gates

- Training candidates: `6000`.
- Validation candidates: `1500`.
- Untouched test tasks: `1000`.
- Updates per learner seed: `3000`.
- Primary deterministic-contract gate: `True`.
- Learned safety gate: `True`.
- Hybrid safety/cost gate: `False`.
- Learned policy reported without test retuning: `True`.

A failed learned or hybrid gate is a valid result; the deterministic controller remains the reference.

## Source hashes

- `experiments/crucible_c1i_learned_applicability_20261021/PROTOCOL.md`: `714153692fcfa8838043017125d9b13a3fcdffea7cc078e1ee6c2af573a401f5`
- `tools/run_crucible_c1h_heldout_generalization_20261021.py`: `adc3f291a3fbaeca8eab62f7521a1b8ab0de6b32d5fb5c2331333f221e64a36d`
- `tools/run_crucible_c1i_learned_applicability_20261021.py`: `85170fbc77c499629f222bb7f7636b5a1a590c958631644f0925270d6cf95766`
