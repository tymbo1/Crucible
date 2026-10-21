# Recursive reasoning controller C0/C1 results

Date: 2026-10-21
Execution boundary: CPU-only; no model, CUDA, network or cloud execution.

## Verdict

- C0 contract gate: **PASS** (11 invariant checks).
- C1 frozen-task replay: **PASS** for shortcut invariance.
- Policy advantage: **not_established**; adaptive policy was not treated as superior by default.
- Wall time for this CPU run: 1.067 seconds.

C0 contract success and C1 policy advantage are separate claims. The synthetic replay validates implementation contracts and exposes failure modes; it does not establish cognitive superiority or future competence.

## C0 contract checks

| Check | Result |
| --- | --- |
| `budget_inheritance_is_monotonic` | PASS |
| `typed_state_hash_round_trip` | PASS |
| `calibrated_entropy_only` | PASS |
| `invalid_belief_rejected` | PASS |
| `cycle_detection` | PASS |
| `budget_exhaustion_is_fail_closed` | PASS |
| `transition_hash_linkage` | PASS |
| `transition_records_are_immutable` | PASS |
| `transactional_memory_commit_and_rollback` | PASS |
| `contradiction_provenance_gate` | PASS |
| `independent_verification` | PASS |

## Manifest and C1 protocol

- Manifest validator oracle checks: 640
- Records replayed from frozen test split: 256
- Policies: one_pass, fixed_depth_2, confidence_stop, verification_only, adaptive
- Shared remaining budget: `{"actions": 8, "cost": 12.0, "counterexample_calls": 2, "depth": 8, "evidence_calls": 2, "memory_writes": 2}`
- Future-competence term: disabled for C0/C1.
- Candidate responses include clean, detectable-error, misleading-evidence, contradiction and shortcut-probe cases.

## C1 policy results

| Policy | Accuracy | Initial accuracy | Corrections | False corrections | Unresolved | Valid memory commits | Cost units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `one_pass` | 0.4023 | 0.4023 | 0 | 0 | 153 | 0 | 256.000 |
| `fixed_depth_2` | 0.7891 | 0.4023 | 99 | 0 | 54 | 0 | 1152.000 |
| `confidence_stop` | 0.4023 | 0.4023 | 0 | 0 | 153 | 0 | 454.000 |
| `verification_only` | 1.0000 | 0.4023 | 153 | 0 | 0 | 256 | 1280.000 |
| `adaptive` | 0.8008 | 0.4023 | 102 | 0 | 51 | 103 | 1701.500 |

## Required limitations

- The candidate engine is deterministic and synthetic; it is not a language model.
- The exact task oracle is used only by the independent verifier and evaluator, not by ordinary policy selection.
- Confidence values are intentionally imperfect fixtures; calibration metrics are diagnostic and do not authorize entropy optimization.
- No calibrated likelihood transition was supplied, so information gain was recorded as unavailable rather than fabricated.
- The future-competence objective was disabled and no learning intervention occurred.
- C1 results do not justify C2 model evaluation, neural training or a claim that adaptive control is beneficial.

## Artifact hashes

- `experiments/crucible_recursive_controller_c0_c1_20261021/C0_RESULTS.json`: `c0f424c600cc91be693ce12e01fc5f1cd9d4d3ff882c871a12ce98e5c7afc309`
- `experiments/crucible_recursive_controller_c0_c1_20261021/C1_RESULTS.json`: `10494f7aee710cb3f6c5cb7de62cac1de255db37dd365f19b9d87e1784aff434`
- `experiments/crucible_recursive_controller_c0_c1_20261021/C1_TRACES.jsonl`: `b325c142f734523d088e1ab8533f45cc17fcd7fbce10d31b40921a48a8d36e98`
- `experiments/crucible_recursive_controller_c0_c1_20261021/MANIFEST_VALIDATION.json`: `10f355085723e4bc5f57b9e0fbf513b8399af1872de19571f409c0623f80da78`
- `experiments/crucible_recursive_controller_c0_c1_20261021/RUN_CONFIG.json`: `05248f08cc67a134e658dfb70af300716b14b2b328ec54de8e999a26ce97cdc2`
- `experiments/crucible_recursive_controller_c0_c1_20261021/SOURCE_HASHES.json`: `39f362e8e1669bb91bb3bf2cba67094bda65b6b5b4e66b4324a892874e6cf213`

## Governing source hashes

- `docs/CRUCIBLE_RECURSIVE_CONTROLLER_OPERATIONAL_SPEC_20261021.md`: `c2ad9a6c96bf98bf5d3f3ae32cba64c1fe449430ed6c7a72c4e0a3485eebb8c6`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/manifest.json`: `d8487034ec4f389c0a9364249757a5e14f0ffbb02000cdc5c10765b58e1acbe9`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/manifest.sha256`: `4b332f06770fd0a6cbb413542d9eb68b205fb1596905dc324ab74a5d88f15332`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/test.jsonl`: `bee94a013c3708b9835b11e68e68d1f346416b2f09d9a7ee58111e3787a7247e`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/train.jsonl`: `a745373304ee52c1422150f47b6e559b6d47ff7c5322b214888b0dd8f144ba35`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/validation.jsonl`: `52c86a67724f9680c20e7263d8ce9f35450bfdbf3fdf2fc8f9a7bed595279650`
- `tools/recursive_controller.py`: `682d9ff779c5f2b5c7b6f078006e741abda36009dcc37f373489b62a3e7ee805`
- `tools/run_recursive_controller_c0_c1.py`: `fd267e601913f0d0677e9c7d0d4db818e99cedcd29ff2d5a1b5c865c442a4d9c`
- `tools/test_recursive_controller.py`: `e5b08c56719239e0333c06d795b584e24d7e3c801e6c38b6621cbff8d0846697`
- `tools/validate_cognitive_manifests.py`: `f6d6c42daf06cd42926fb29bfb6de059d076ee73a0b06692f960699da5527359`
