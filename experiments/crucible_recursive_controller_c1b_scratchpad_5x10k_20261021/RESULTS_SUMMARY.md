# C1b persistent scratchpad continuation

Date: 2026-10-21
Execution: CPU-only; five sequential 10,000-update phases per seed.

## Verdict

This is a small policy-learner continuation experiment. It does not establish neural memory, cumulative language learning or general intelligence.
- Seeds: **3/3**.
- Phases per arm and seed: **5**.
- Updates per phase: **10000**.
- Total continuation updates per arm and seed: **50000**.

## Final untouched-test comparison

| Arm | Accuracy | Mean cost | False corrections | Unresolved | Cost/accuracy score | Scratchpad hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_scratchpad` | 0.7135 | 5.365 | 0.00 | 73.33 | 0.6912 | 0.0000 |
| `persistent_scratchpad` | 0.7344 | 5.488 | 0.00 | 68.00 | 0.7115 | 0.0000 |

## Validation progression

| Seed | Arm | Phase | Accuracy | Mean cost | Score | Scratchpad entries | Mature entries | Skipped updates |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `no_scratchpad` | 1 | 0.7812 | 4.309 | 0.7633 | 0 | 0 | 0 |
| 42 | `no_scratchpad` | 2 | 0.7812 | 4.309 | 0.7633 | 0 | 0 | 0 |
| 42 | `no_scratchpad` | 3 | 0.7812 | 4.730 | 0.7615 | 0 | 0 | 0 |
| 42 | `no_scratchpad` | 4 | 0.7812 | 4.281 | 0.7634 | 0 | 0 | 0 |
| 42 | `no_scratchpad` | 5 | 0.7812 | 4.258 | 0.7635 | 0 | 0 | 0 |
| 42 | `persistent_scratchpad` | 1 | 0.7812 | 4.309 | 0.7633 | 185 | 3 | 107 |
| 42 | `persistent_scratchpad` | 2 | 0.7812 | 4.543 | 0.7623 | 185 | 3 | 100 |
| 42 | `persistent_scratchpad` | 3 | 0.7812 | 4.730 | 0.7615 | 185 | 4 | 103 |
| 42 | `persistent_scratchpad` | 4 | 0.7812 | 4.309 | 0.7633 | 185 | 2 | 74 |
| 42 | `persistent_scratchpad` | 5 | 0.7656 | 4.816 | 0.7456 | 185 | 2 | 73 |
| 1337 | `no_scratchpad` | 1 | 0.8125 | 4.309 | 0.7945 | 0 | 0 | 0 |
| 1337 | `no_scratchpad` | 2 | 0.8125 | 4.355 | 0.7944 | 0 | 0 | 0 |
| 1337 | `no_scratchpad` | 3 | 0.8125 | 4.355 | 0.7944 | 0 | 0 | 0 |
| 1337 | `no_scratchpad` | 4 | 0.8125 | 4.699 | 0.7929 | 0 | 0 | 0 |
| 1337 | `no_scratchpad` | 5 | 0.8125 | 4.293 | 0.7946 | 0 | 0 | 0 |
| 1337 | `persistent_scratchpad` | 1 | 0.8125 | 4.309 | 0.7945 | 186 | 13 | 440 |
| 1337 | `persistent_scratchpad` | 2 | 0.8125 | 4.480 | 0.7938 | 186 | 11 | 542 |
| 1337 | `persistent_scratchpad` | 3 | 0.8125 | 4.160 | 0.7952 | 186 | 8 | 372 |
| 1337 | `persistent_scratchpad` | 4 | 0.8125 | 4.762 | 0.7927 | 186 | 3 | 165 |
| 1337 | `persistent_scratchpad` | 5 | 0.7969 | 4.309 | 0.7789 | 186 | 2 | 94 |
| 2026 | `no_scratchpad` | 1 | 0.7500 | 4.082 | 0.7330 | 0 | 0 | 0 |
| 2026 | `no_scratchpad` | 2 | 0.7734 | 4.684 | 0.7539 | 0 | 0 | 0 |
| 2026 | `no_scratchpad` | 3 | 0.7344 | 3.664 | 0.7191 | 0 | 0 | 0 |
| 2026 | `no_scratchpad` | 4 | 0.7344 | 3.594 | 0.7194 | 0 | 0 | 0 |
| 2026 | `no_scratchpad` | 5 | 0.7734 | 4.734 | 0.7537 | 0 | 0 | 0 |
| 2026 | `persistent_scratchpad` | 1 | 0.7344 | 3.258 | 0.7208 | 181 | 23 | 620 |
| 2026 | `persistent_scratchpad` | 2 | 0.7734 | 4.211 | 0.7559 | 181 | 23 | 964 |
| 2026 | `persistent_scratchpad` | 3 | 0.7500 | 3.832 | 0.7340 | 181 | 17 | 698 |
| 2026 | `persistent_scratchpad` | 4 | 0.7344 | 3.508 | 0.7198 | 181 | 13 | 506 |
| 2026 | `persistent_scratchpad` | 5 | 0.7734 | 4.188 | 0.7560 | 181 | 12 | 439 |

## Starting models

Source: `experiments/crucible_recursive_controller_c1b_l100000_20261021/RESULTS.json`; source SHA-256: `77cf5daf1d6ed8016858193d87968bacd374a8960f91627789503487ea0f16d0`.
The three starts are the validation-selected C1b-L100000 learner snapshots; prior artifacts were not modified.

| Seed | Selected checkpoint | Snapshot SHA-256 |
| ---: | ---: | --- |
| 42 | 5000 | `8173b12e51f8e8e7448cfd88045e3565f93833b4e9626a8044d9dc32fa29f59c` |
| 1337 | 50000 | `72f917d4b3418de59964680900e8b9aaf9a9c9d1177b0c48d0cfe65aaba29e0b` |
| 2026 | 25000 | `712f14f535e94133fb4e79f6903b85b7189f1fb034d524d0f7722d7209359a77` |

## Scratchpad interpretation

The scratchpad records only decision-time feature keys, selected actions and observed training rewards. It is read-only during validation and test evaluation. Mature associations can bypass a stable redundant model update; materially different outcomes reopen the association.

## Limitations

This experiment uses the synthetic C1b corpus and a linear contextual bandit. A scratchpad hit is not proof that the underlying procedure is correct, and reduced updates are not equivalent to improved neural learning. The result does not transfer automatically to the 170M Crucible model or CMS2.

## Source hashes

- `experiments/crucible_recursive_controller_c1b_l100000_20261021/HASHES.json`: `8adc0bda1eca0e5d38c9b8171e3fd33943337c7341559f3f256eafbcd9c36451`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/RESULTS.json`: `77cf5daf1d6ed8016858193d87968bacd374a8960f91627789503487ea0f16d0`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/test.jsonl`: `36bcd857986bdb42d26db5fce47385d9b6575634c49edbf58fcac40cae12bd79`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/train.jsonl`: `a82aa634baeb4cacb20efc663b14b0ecc6696271ba7a8e1a12723480f3dd2698`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/validation.jsonl`: `caf54c4ce7dc9380423229da9889c4bc4c3f400d2fb9003b13f4bf5beb648f0a`
- `experiments/crucible_recursive_controller_c1b_scratchpad_5x10k_20261021/PROTOCOL.md`: `50c3b8af0c56c92e3e8c2cb13ecb6b56379aa97e354a6d2ae95ca4199983150c`
- `tools/cognitive_task_suite.py`: `639274386a3f1f78dc35be61aa737e170c39f008e07828cbd04506a22652d781`
- `tools/recursive_controller.py`: `6df88d6b6d16de6ac79d378efcdaf06abe2a3d74152dcc39f2f147026e5183bf`
- `tools/run_recursive_controller_c1b_l3000.py`: `166c171dd48f6feac42a9805d1539801874e649381744c21479b6de0bd3cec64`
- `tools/run_recursive_controller_c1b_scratchpad_5x10k.py`: `9bf6627512ab6db54223258db454de4deb37633128513e134370779232c2d1b6`
