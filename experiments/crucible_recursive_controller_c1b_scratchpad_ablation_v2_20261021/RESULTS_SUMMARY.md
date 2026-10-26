# C1b scratchpad factorial ablation

Date: 2026-10-21
Execution: CPU-only; four paired arms, five sequential 10,000-update phases.

## Final untouched-test comparison

| Arm | Accuracy | Mean cost | False corrections | Unresolved | Score | Scratchpad hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.7135 | 5.365 | 0.00 | 73.33 | 0.6912 | 0.0000 |
| `memory_only` | 0.7057 | 5.311 | 0.00 | 75.33 | 0.6836 | 0.0000 |
| `skip_only` | 0.7057 | 5.347 | 0.00 | 75.33 | 0.6834 | 0.0000 |
| `full` | 0.7344 | 5.488 | 0.00 | 68.00 | 0.7115 | 0.0000 |

Memory arms were also evaluated with scratchpad access disabled after phase five; those read-only comparisons are stored in `RESULTS.json`.

## Validation progression

| Seed | Arm | Phase | Accuracy | Mean cost | Score | Applied updates | Skipped updates | Mature entries |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `baseline` | 1 | 0.7812 | 4.309 | 0.7633 | 10000 | 0 | 0 |
| 42 | `baseline` | 2 | 0.7812 | 4.309 | 0.7633 | 10000 | 0 | 0 |
| 42 | `baseline` | 3 | 0.7812 | 4.730 | 0.7615 | 10000 | 0 | 0 |
| 42 | `baseline` | 4 | 0.7812 | 4.281 | 0.7634 | 10000 | 0 | 0 |
| 42 | `baseline` | 5 | 0.7812 | 4.258 | 0.7635 | 10000 | 0 | 0 |
| 42 | `memory_only` | 1 | 0.7812 | 4.309 | 0.7633 | 10000 | 0 | 3 |
| 42 | `memory_only` | 2 | 0.7812 | 4.621 | 0.7620 | 10000 | 0 | 5 |
| 42 | `memory_only` | 3 | 0.7812 | 4.992 | 0.7604 | 10000 | 0 | 5 |
| 42 | `memory_only` | 4 | 0.7812 | 5.121 | 0.7599 | 10000 | 0 | 2 |
| 42 | `memory_only` | 5 | 0.7812 | 4.324 | 0.7632 | 10000 | 0 | 2 |
| 42 | `skip_only` | 1 | 0.7734 | 4.035 | 0.7566 | 9986 | 14 | 0 |
| 42 | `skip_only` | 2 | 0.7812 | 4.402 | 0.7629 | 10000 | 0 | 2 |
| 42 | `skip_only` | 3 | 0.7812 | 4.516 | 0.7624 | 9999 | 1 | 2 |
| 42 | `skip_only` | 4 | 0.7812 | 4.852 | 0.7610 | 10000 | 0 | 1 |
| 42 | `skip_only` | 5 | 0.7812 | 4.352 | 0.7631 | 10000 | 0 | 1 |
| 42 | `full` | 1 | 0.7812 | 4.309 | 0.7633 | 9893 | 107 | 3 |
| 42 | `full` | 2 | 0.7812 | 4.543 | 0.7623 | 9900 | 100 | 3 |
| 42 | `full` | 3 | 0.7812 | 4.730 | 0.7615 | 9897 | 103 | 4 |
| 42 | `full` | 4 | 0.7812 | 4.309 | 0.7633 | 9926 | 74 | 2 |
| 42 | `full` | 5 | 0.7656 | 4.816 | 0.7456 | 9927 | 73 | 2 |
| 1337 | `baseline` | 1 | 0.8125 | 4.309 | 0.7945 | 10000 | 0 | 0 |
| 1337 | `baseline` | 2 | 0.8125 | 4.355 | 0.7944 | 10000 | 0 | 0 |
| 1337 | `baseline` | 3 | 0.8125 | 4.355 | 0.7944 | 10000 | 0 | 0 |
| 1337 | `baseline` | 4 | 0.8125 | 4.699 | 0.7929 | 10000 | 0 | 0 |
| 1337 | `baseline` | 5 | 0.8125 | 4.293 | 0.7946 | 10000 | 0 | 0 |
| 1337 | `memory_only` | 1 | 0.8125 | 4.309 | 0.7945 | 10000 | 0 | 13 |
| 1337 | `memory_only` | 2 | 0.8125 | 4.387 | 0.7942 | 10000 | 0 | 13 |
| 1337 | `memory_only` | 3 | 0.8125 | 4.355 | 0.7944 | 10000 | 0 | 10 |
| 1337 | `memory_only` | 4 | 0.8125 | 4.762 | 0.7927 | 10000 | 0 | 6 |
| 1337 | `memory_only` | 5 | 0.8125 | 4.762 | 0.7927 | 10000 | 0 | 5 |
| 1337 | `skip_only` | 1 | 0.8125 | 4.309 | 0.7945 | 9682 | 318 | 10 |
| 1337 | `skip_only` | 2 | 0.8125 | 4.355 | 0.7944 | 9593 | 407 | 8 |
| 1337 | `skip_only` | 3 | 0.8125 | 4.355 | 0.7944 | 9702 | 298 | 7 |
| 1337 | `skip_only` | 4 | 0.8125 | 4.855 | 0.7923 | 9857 | 143 | 3 |
| 1337 | `skip_only` | 5 | 0.8125 | 4.418 | 0.7941 | 9945 | 55 | 2 |
| 1337 | `full` | 1 | 0.8125 | 4.309 | 0.7945 | 9560 | 440 | 13 |
| 1337 | `full` | 2 | 0.8125 | 4.480 | 0.7938 | 9458 | 542 | 11 |
| 1337 | `full` | 3 | 0.8125 | 4.160 | 0.7952 | 9628 | 372 | 8 |
| 1337 | `full` | 4 | 0.8125 | 4.762 | 0.7927 | 9835 | 165 | 3 |
| 1337 | `full` | 5 | 0.7969 | 4.309 | 0.7789 | 9906 | 94 | 2 |
| 2026 | `baseline` | 1 | 0.7500 | 4.082 | 0.7330 | 10000 | 0 | 0 |
| 2026 | `baseline` | 2 | 0.7734 | 4.684 | 0.7539 | 10000 | 0 | 0 |
| 2026 | `baseline` | 3 | 0.7344 | 3.664 | 0.7191 | 10000 | 0 | 0 |
| 2026 | `baseline` | 4 | 0.7344 | 3.594 | 0.7194 | 10000 | 0 | 0 |
| 2026 | `baseline` | 5 | 0.7734 | 4.734 | 0.7537 | 10000 | 0 | 0 |
| 2026 | `memory_only` | 1 | 0.7344 | 3.371 | 0.7203 | 10000 | 0 | 22 |
| 2026 | `memory_only` | 2 | 0.7734 | 4.086 | 0.7564 | 10000 | 0 | 20 |
| 2026 | `memory_only` | 3 | 0.7734 | 3.828 | 0.7575 | 10000 | 0 | 14 |
| 2026 | `memory_only` | 4 | 0.7344 | 3.828 | 0.7184 | 10000 | 0 | 11 |
| 2026 | `memory_only` | 5 | 0.7734 | 4.008 | 0.7567 | 10000 | 0 | 10 |
| 2026 | `skip_only` | 1 | 0.7109 | 3.422 | 0.6967 | 9612 | 388 | 14 |
| 2026 | `skip_only` | 2 | 0.7734 | 4.711 | 0.7538 | 9259 | 741 | 16 |
| 2026 | `skip_only` | 3 | 0.7266 | 3.730 | 0.7110 | 9471 | 529 | 13 |
| 2026 | `skip_only` | 4 | 0.7109 | 3.574 | 0.6960 | 9603 | 397 | 11 |
| 2026 | `skip_only` | 5 | 0.7734 | 4.570 | 0.7544 | 9652 | 348 | 10 |
| 2026 | `full` | 1 | 0.7344 | 3.258 | 0.7208 | 9380 | 620 | 23 |
| 2026 | `full` | 2 | 0.7734 | 4.211 | 0.7559 | 9036 | 964 | 23 |
| 2026 | `full` | 3 | 0.7500 | 3.832 | 0.7340 | 9302 | 698 | 17 |
| 2026 | `full` | 4 | 0.7344 | 3.508 | 0.7198 | 9494 | 506 | 13 |
| 2026 | `full` | 5 | 0.7734 | 4.188 | 0.7560 | 9561 | 439 | 12 |

## Interpretation

`memory_only` isolates accessible retained associations. `skip_only` uses the same private redundancy detector as `full` but cannot use stored associations to choose actions. `full` combines both mechanisms. This is a synthetic linear-bandit result, not neural memory or cumulative language learning.

## Source hashes

- `experiments/crucible_recursive_controller_c1b_l100000_20261021/HASHES.json`: `8adc0bda1eca0e5d38c9b8171e3fd33943337c7341559f3f256eafbcd9c36451`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/RESULTS.json`: `77cf5daf1d6ed8016858193d87968bacd374a8960f91627789503487ea0f16d0`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/test.jsonl`: `36bcd857986bdb42d26db5fce47385d9b6575634c49edbf58fcac40cae12bd79`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/train.jsonl`: `a82aa634baeb4cacb20efc663b14b0ecc6696271ba7a8e1a12723480f3dd2698`
- `experiments/crucible_recursive_controller_c1b_l100000_20261021/manifests/validation.jsonl`: `caf54c4ce7dc9380423229da9889c4bc4c3f400d2fb9003b13f4bf5beb648f0a`
- `experiments/crucible_recursive_controller_c1b_scratchpad_ablation_v2_20261021/PROTOCOL.md`: `59e68ca53bdb56530f01d59aff00f600583b3d5a4dfd5d7d9b560dbb268efb57`
- `tools/cognitive_task_suite.py`: `639274386a3f1f78dc35be61aa737e170c39f008e07828cbd04506a22652d781`
- `tools/recursive_controller.py`: `6df88d6b6d16de6ac79d378efcdaf06abe2a3d74152dcc39f2f147026e5183bf`
- `tools/run_recursive_controller_c1b_l3000.py`: `166c171dd48f6feac42a9805d1539801874e649381744c21479b6de0bd3cec64`
- `tools/run_recursive_controller_c1b_scratchpad_5x10k.py`: `9bf6627512ab6db54223258db454de4deb37633128513e134370779232c2d1b6`
- `tools/run_recursive_controller_c1b_scratchpad_ablation_20261021.py`: `758d9a0fd6cfb34e520f7f8258c4fd34ab06662cd97bcd01e9fa22778f827a69`
