# C1b-L3000 learned strategy-selection pilot

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud, model training or checkpoint download.

## Verdict

This is a bounded synthetic feasibility experiment. It does not establish cognitive superiority, metacognition or future competence.
- Seeds completed: **3/3**.
- Policy updates per seed: **3000**.
- Wall time: **1.496 s**; cap: **1200.0 s**.
- Sealed-C1 input overlap: **0**.

## Test results by learner seed

The test split was evaluated after validation-based checkpoint selection. Test outcomes did not select a checkpoint.

| Seed | Policy | Accuracy | Mean cost | False corrections | Unresolved | Verification omissions | Cost/accuracy score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `verification_only` | 0.7539 | 5.451 | 0 | 63 | 0 | 0.7312 |
| 42 | `verification_first_adaptive` | 0.6562 | 5.062 | 0 | 88 | 91 | 0.6352 |
| 42 | `adaptive` | 0.6680 | 5.316 | 0 | 85 | 122 | 0.6458 |
| 42 | `fixed_depth_2` | 0.5781 | 3.500 | 0 | 108 | 256 | 0.5635 |
| 42 | `learned_adaptive` | 0.6758 | 5.146 | 0 | 83 | 48 | 0.6543 |
| 1337 | `verification_only` | 0.7891 | 5.703 | 0 | 54 | 0 | 0.7653 |
| 1337 | `verification_first_adaptive` | 0.6992 | 5.020 | 0 | 77 | 95 | 0.6783 |
| 1337 | `adaptive` | 0.6719 | 5.484 | 0 | 84 | 122 | 0.6490 |
| 1337 | `fixed_depth_2` | 0.5781 | 3.500 | 0 | 108 | 256 | 0.5635 |
| 1337 | `learned_adaptive` | 0.7500 | 5.367 | 0 | 64 | 49 | 0.7276 |
| 2026 | `verification_only` | 0.7578 | 5.627 | 0 | 62 | 0 | 0.7344 |
| 2026 | `verification_first_adaptive` | 0.6797 | 5.051 | 0 | 82 | 94 | 0.6586 |
| 2026 | `adaptive` | 0.6406 | 5.344 | 0 | 92 | 122 | 0.6184 |
| 2026 | `fixed_depth_2` | 0.5781 | 3.500 | 0 | 108 | 256 | 0.5635 |
| 2026 | `learned_adaptive` | 0.8086 | 5.818 | 0 | 49 | 49 | 0.7844 |

## Historical C1 reference

The sealed C1 replay reported verification-only at 100% accuracy / mean cost 5.0 and adaptive at 80.08% / mean cost 6.646. Those cases are not reused here; the values are historical context only.

## Experimental contract

- New task seed: `20261022`; generated records: `{"test": 256, "train": 256, "validation": 128}`.
- Training, validation and test record IDs are disjoint; all generated input hashes were checked against the sealed C1 manifests.
- Training conditions were R/I/U/A; test conditions included R/I/E/D/U/A with declared and inferred-shift modes.
- Latent answers were evaluator-only. The learned policy received state features and declared tool metadata only.
- The bandit selected the first operation; a deterministic bounded continuation executed it. No Q-learning or neural policy was run.
- Validation score: accuracy minus preregistered cost, false-correction and unresolved penalties; ties selected the earliest checkpoint.
- Future-competence objective: disabled.

## Limitations and decision gate

A first-operation contextual bandit cannot infer an undisclosed environment shift before observing a tool result. Therefore inferred-shift performance is a deliberate limitation of this pilot, not evidence of sequential adaptation.
The learned policy only earns a follow-on sequential-controller experiment if it beats the best frozen rule on a held-out declared shift without a safety regression. Otherwise the deterministic rule remains preferred.

## Governing source hashes

- `experiments/crucible_cognitive_eval_v2_20261021/manifests/test.jsonl`: `bee94a013c3708b9835b11e68e68d1f346416b2f09d9a7ee58111e3787a7247e`
- `experiments/crucible_recursive_controller_c1b_20261021/C1B_L3000_PROTOCOL.md`: `ff1e511a7b16c25b3120fd14afa869b7ccc6396b790f17514c05db4f6ed17a83`
- `experiments/crucible_recursive_controller_c1b_20261021/PROTOCOL_V2.md`: `e5917ab3a9b54d82f403105d3832cbd9fda883e34d6a392b8a74b47b0da6beb8`
- `tools/cognitive_task_suite.py`: `639274386a3f1f78dc35be61aa737e170c39f008e07828cbd04506a22652d781`
- `tools/recursive_controller.py`: `6df88d6b6d16de6ac79d378efcdaf06abe2a3d74152dcc39f2f147026e5183bf`
- `tools/run_recursive_controller_c1b_l3000.py`: `166c171dd48f6feac42a9805d1539801874e649381744c21479b6de0bd3cec64`
- `tools/validate_cognitive_manifests.py`: `f6d6c42daf06cd42926fb29bfb6de059d076ee73a0b06692f960699da5527359`
