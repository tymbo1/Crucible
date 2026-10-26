# C1f acquire, retain, adapt and reuse

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Verdict

This is a sequential controller and memory-contract experiment, not a neural-model capability result.

## Aggregate results

| Arm | Accuracy | Novel accuracy | Later accuracy | Mean cost | Retrieval relevance | Adaptation success | Harmful transfer | Final memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 0.7500 | 0.7500 | 0.7500 | 0.712 | 0.2428 | 0.0833 | 0.0000 | 1.5 |
| `adaptive_memory_disabled` | 0.8333 | 0.8333 | 0.8333 | 1.700 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| `adaptive_shuffled_memory` | 0.7917 | 0.7917 | 0.7917 | 0.548 | 0.6146 | 0.0000 | 0.1250 | 6.0 |
| `oracle` | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| `persistent` | 0.8333 | 0.8333 | 0.8333 | 0.669 | 0.5000 | 0.0000 | 0.0000 | 3.0 |
| `stateless` | 0.8333 | 0.8333 | 0.8333 | 1.700 | 0.0000 | 0.0000 | 0.0000 | 0.0 |

## Environment results

### informative

| Arm | Accuracy | Later accuracy | Mean cost | Harmful transfer |
| --- | ---: | ---: | ---: | ---: |
| `adaptive` | 1.0000 | 1.0000 | 0.549 | 0.0000 |
| `adaptive_memory_disabled` | 1.0000 | 1.0000 | 1.000 | 0.0000 |
| `adaptive_shuffled_memory` | 1.0000 | 1.0000 | 0.542 | 0.0000 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 |
| `persistent` | 1.0000 | 1.0000 | 0.083 | 0.0000 |
| `stateless` | 1.0000 | 1.0000 | 1.000 | 0.0000 |

### expensive

| Arm | Accuracy | Later accuracy | Mean cost | Harmful transfer |
| --- | ---: | ---: | ---: | ---: |
| `adaptive` | 0.6667 | 0.6667 | 0.000 | 0.0000 |
| `adaptive_memory_disabled` | 1.0000 | 1.0000 | 3.500 | 0.0000 |
| `adaptive_shuffled_memory` | 0.8333 | 0.8333 | 0.500 | 0.0000 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 |
| `persistent` | 1.0000 | 1.0000 | 0.292 | 0.0000 |
| `stateless` | 1.0000 | 1.0000 | 3.500 | 0.0000 |

### noisy

| Arm | Accuracy | Later accuracy | Mean cost | Harmful transfer |
| --- | ---: | ---: | ---: | ---: |
| `adaptive` | 0.6667 | 0.6667 | 1.300 | 0.0000 |
| `adaptive_memory_disabled` | 0.6667 | 0.6667 | 1.300 | 0.0000 |
| `adaptive_shuffled_memory` | 0.8333 | 0.8333 | 1.150 | 0.0000 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 |
| `persistent` | 0.6667 | 0.6667 | 1.300 | 0.0000 |
| `stateless` | 0.6667 | 0.6667 | 1.300 | 0.0000 |

### unresolvable

| Arm | Accuracy | Later accuracy | Mean cost | Harmful transfer |
| --- | ---: | ---: | ---: | ---: |
| `adaptive` | 0.6667 | 0.6667 | 1.000 | 0.0000 |
| `adaptive_memory_disabled` | 0.6667 | 0.6667 | 1.000 | 0.0000 |
| `adaptive_shuffled_memory` | 0.5000 | 0.5000 | 0.000 | 0.5000 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 |
| `persistent` | 0.6667 | 0.6667 | 1.000 | 0.0000 |
| `stateless` | 0.6667 | 0.6667 | 1.000 | 0.0000 |

## Preregistered gates

- Accuracy/cost transfer gate: `True`.
- Adaptive harmful-transfer gate: `True`.
- Shuffled-memory harmful-transfer gate: `False`.
- The shuffled-memory and memory-disabled controls are included to separate genuine reuse from incidental computation or contamination.

## Provenance

- Three seeds: 42, 1337 and 2026.
- 12 novel episodes per family, variant and environment per seed.
- Every retained record contains rule, conditions, evidence, provenance and validation status.
- No task answer or hidden rule was provided as a policy feature.

## Source hashes

- `experiments/crucible_c1f_persistent_reuse_20261021/PROTOCOL.md`: `068c6735c3390c4b08e0c5685583fc6fe7cbe28fdfec02c658f522763a6f777c`
- `tools/run_crucible_c1f_persistent_reuse_20261021.py`: `8a8af7d4a12e4b9fd124331957ef688867915857f9136e86dc4d41d1dd3f2343`
