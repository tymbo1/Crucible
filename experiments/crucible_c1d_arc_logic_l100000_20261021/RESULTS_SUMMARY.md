# C1d-L100000 logic, arithmetic and ARC strategy selection

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible model training.

## Verdict

- Seeds: **3/3**; updates per seed: **100000**.
- Wall time: **11.160 s**; cap: **1200.0 s**.
- The public ARC evaluation split was used only after validation checkpoint selection.
- This tests solver selection, not general intelligence or neural reasoning.

## Final test results

| Policy | All tasks | ARC public evaluation | Logic | Math | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `first_consistent` | 0.4510 | 0.0025 | 1.0000 | 0.5000 | 0.621 |
| `cheapest_consistent` | 0.4510 | 0.0025 | 1.0000 | 0.5000 | 0.621 |
| `family_heuristic` | 0.4510 | 0.0025 | 1.0000 | 0.5000 | 0.831 |
| `random_consistent` | 0.4107 | 0.0025 | 1.0000 | 0.3656 | 0.841 |
| `oracle` | 0.6010 | 0.0025 | 1.0000 | 1.0000 | 0.651 |
| `learned` | 0.4510 | 0.0025 | 1.0000 | 0.5000 | 0.621 |

## Data separation

- ARC source: [official ARC-AGI-1 repository](https://github.com/fchollet/ARC-AGI), commit `399030444e0ab0cc8b4e199870fb20b863846f34`.
- Problem counts: `{"arc_public_evaluation": 400, "test": 1000, "train": 1520, "validation": 480}`.
- ARC training tasks were split into train/validation; ARC public evaluation tasks were not used for training or checkpoint selection.
- Generated logic/math tasks use independent seeds and disjoint IDs.
- Held-out outputs were evaluator-only.

## Checkpoint selection

- Seed 42: selected update `0`; validation score `0.625433`.
- Seed 1337: selected update `0`; validation score `0.625433`.
- Seed 2026: selected update `0`; validation score `0.625433`.

## Limitations

The solver portfolio is intentionally small. A low ARC score can mean that no included transformation generated the correct output, not that the selector failed to choose among available correct candidates. ARC results are public-evaluation measurements, not private leaderboard results.

## Source hashes

- `experiments/crucible_c1d_arc_logic_l100000_20261021/PROTOCOL.md`: `00324d85e95dfca75e855296028f287cc576aa995babdbb709fdc71289586f25`
- `tools/run_crucible_c1d_arc_logic_l100000.py`: `549e5c26d73630ca2dc25cd9ce31c2466eb5826b8e66ab4e043d567a9ee865ba`
