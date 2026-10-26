# C1e ARC portfolio expansion and math identifiability

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Verdict

- Track A measures solver capability before any controller retraining.
- Track B measures the observable ceiling of the frozen C1d math solver choice.
- The C1d-L100000 experiment and artifacts remain unchanged.

## Track A — ARC public evaluation

| Search level | Programs | Demonstration coverage | Portfolio oracle accuracy | Mean candidates |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 10 | 0.0025 | 0.0025 | 10.0 |
| `objects` | 31 | 0.0025 | 0.0025 | 31.0 |
| `compose2` | 814 | 0.0025 | 0.0025 | 814.0 |

The oracle only counts exact programs that fit every demonstration. No selector was trained in this track.

## Track B — frozen math identifiability

- Frozen solver oracle accuracy: **1.0000**.
- C1d-feature observable upper bound: **0.5000**.
- Demonstration-signature observable upper bound: **0.5000**.
- Semantics counts: `{"left": 150, "precedence": 150}`.
- The generated demonstrations expose only addition and subtraction; multiplication is introduced only in held-out cases. This makes left-versus-precedence selection unidentifiable from the supplied demonstrations.

## Decision

Do not increase controller capacity or run another 100k-step selector until the ARC portfolio oracle rises and the math task generator exposes the strategy distinction it asks the controller to learn.

## Provenance

- ARC source: [official ARC-AGI-1 repository](https://github.com/fchollet/ARC-AGI), commit `399030444e0ab0cc8b4e199870fb20b863846f34`.
- ARC counts: `{"evaluation": 400, "train": 320, "validation": 80}`.
- Public ARC outputs were used only for final portfolio scoring; no learned checkpoint or selector was selected from them.

## Source hashes

- `experiments/crucible_c1d_arc_logic_l100000_20261021/HASHES.json`: `8141b3b5408aad70e874218fb67b342c93901e2ae80bd28070853442132a0c06`
- `experiments/crucible_c1d_arc_logic_l100000_20261021/RESULTS_SUMMARY.md`: `f279a6ad77014bf9a6ac10358ace9e08b7846bdca1ea3e78579b27823748ad6e`
- `experiments/crucible_c1e_arc_portfolio_math_20261021/PROTOCOL.md`: `f733743995065f461eb671f172b647273e16cfa180166f5513cfc49356f4543c`
- `tools/run_crucible_c1e_arc_portfolio_math_20261021.py`: `bd1d1e3348c6553fb4dcd76f79de4c37f3a3c7bc432f5b66abef72279d0d39d6`
