# C1l verification amortization and certificate invalidation

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate paired results

| Policy | Accuracy | Mean task cost | Mean cost incl. shared issuance | Harmful transfer | Valid certificate uses | Invalidations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `certificate_amortized` | 0.8737 | 1.100 | 1.102 | 0.0104 | 6921 | 23079 |
| `certificate_corrupted` | 0.8737 | 1.136 | 1.138 | 0.0104 | 4740 | 25260 |
| `deterministic_reference` | 0.8761 | 1.145 | 1.145 | 0.0000 | 0 | 0 |
| `learned_constrained_reference` | 0.8761 | 1.066 | 1.066 | 0.0000 | 0 | 0 |
| `stale_certificate_control` | 0.6107 | 0.066 | 0.068 | 0.3983 | 0 | 0 |
| `stateless_reference` | 0.8761 | 1.210 | 1.210 | 0.0000 | 0 | 0 |

## Certificate catalogue

- Independent seed candidates: `50000`.
- Catalogue entries: `16` / `16`.
- Catalogue complete: `True`.
- Test-task overlap: `0`.

## Cost decomposition

- Certificate lookup cost: `0.02`.
- Certificate validity cost: `0.04`.
- Issuance cost per certificate: `1.98`.
- Shared catalogue issuance cost: `63.36`.

## Invalidation reasons

- `contradictory_record`: `165`.
- `insufficient_evidence_quality`: `228`.
- `invalid_provenance`: `3369`.
- `no_matching_certificate`: `12153`.
- `probe_disagreement`: `7164`.

## Harmful-transfer attribution

### `certificate_amortized` — `72` events
- `by_condition`: `{'adaptation': 72}`.
- `by_family`: `{'tree': 72}`.
- `by_memory_size`: `{'10': 3, '100': 6, '1000': 18, '10000': 45}`.
- `by_decision`: `{'adapt': 72}`.
- `by_source`: `{'certificate': 72}`.
- `by_certificate_status`: `{'valid': 72}`.
### `certificate_corrupted` — `72` events
- `by_condition`: `{'adaptation': 72}`.
- `by_family`: `{'tree': 72}`.
- `by_memory_size`: `{'10': 3, '100': 6, '1000': 18, '10000': 45}`.
- `by_decision`: `{'adapt': 72}`.
- `by_source`: `{'certificate': 72}`.
- `by_certificate_status`: `{'valid': 72}`.
### `stale_certificate_control` — `11550` events
- `by_condition`: `{'adaptation': 282, 'applicable': 282, 'contradictory': 3420, 'invalidated': 3114, 'misleading_similarity': 3534, 'noisy_evidence': 339, 'provenance_corrupt': 288, 'unresolvable': 291}`.
- `by_family`: `{'interval': 2598, 'matrix': 2982, 'multiset': 2907, 'tree': 3063}`.
- `by_memory_size`: `{'10': 2874, '100': 2676, '1000': 2991, '10000': 3009}`.
- `by_decision`: `{'adapt': 4107, 'reuse': 7443}`.
- `by_source`: `{'stale_certificate': 11550}`.
- `by_certificate_status`: `{'stale_used': 11550}`.

## Results by condition

| Condition | Policy | Accuracy | Mean cost | Harmful transfer |
| --- | --- | ---: | ---: | ---: |
| `applicable` | `certificate_amortized` | 1.0000 | 0.219 | 0.0000 |
| `applicable` | `certificate_corrupted` | 1.0000 | 0.260 | 0.0000 |
| `applicable` | `deterministic_reference` | 1.0000 | 0.320 | 0.0000 |
| `applicable` | `learned_constrained_reference` | 1.0000 | 0.240 | 0.0000 |
| `applicable` | `stale_certificate_control` | 0.9234 | 0.066 | 0.0791 |
| `applicable` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 |
| `adaptation` | `certificate_amortized` | 0.9819 | 0.192 | 0.0198 |
| `adaptation` | `certificate_corrupted` | 0.9819 | 0.423 | 0.0198 |
| `adaptation` | `deterministic_reference` | 1.0000 | 0.771 | 0.0000 |
| `adaptation` | `learned_constrained_reference` | 1.0000 | 0.700 | 0.0000 |
| `adaptation` | `stale_certificate_control` | 0.9292 | 0.072 | 0.0733 |
| `adaptation` | `stateless_reference` | 1.0000 | 1.366 | 0.0000 |
| `invalidated` | `certificate_amortized` | 1.0000 | 1.540 | 0.0000 |
| `invalidated` | `certificate_corrupted` | 1.0000 | 1.540 | 0.0000 |
| `invalidated` | `deterministic_reference` | 1.0000 | 1.480 | 0.0000 |
| `invalidated` | `learned_constrained_reference` | 1.0000 | 1.400 | 0.0000 |
| `invalidated` | `stale_certificate_control` | 0.1121 | 0.081 | 0.9260 |
| `invalidated` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 |
| `contradictory` | `certificate_amortized` | 1.0000 | 1.540 | 0.0000 |
| `contradictory` | `certificate_corrupted` | 1.0000 | 1.540 | 0.0000 |
| `contradictory` | `deterministic_reference` | 1.0000 | 1.480 | 0.0000 |
| `contradictory` | `learned_constrained_reference` | 1.0000 | 1.400 | 0.0000 |
| `contradictory` | `stale_certificate_control` | 0.0784 | 0.070 | 0.9540 |
| `contradictory` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 |
| `misleading_similarity` | `certificate_amortized` | 1.0000 | 1.540 | 0.0000 |
| `misleading_similarity` | `certificate_corrupted` | 1.0000 | 1.540 | 0.0000 |
| `misleading_similarity` | `deterministic_reference` | 1.0000 | 1.480 | 0.0000 |
| `misleading_similarity` | `learned_constrained_reference` | 1.0000 | 1.400 | 0.0000 |
| `misleading_similarity` | `stale_certificate_control` | 0.0903 | 0.071 | 0.9424 |
| `misleading_similarity` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 |
| `noisy_evidence` | `certificate_amortized` | 1.0000 | 2.040 | 0.0000 |
| `noisy_evidence` | `certificate_corrupted` | 1.0000 | 2.040 | 0.0000 |
| `noisy_evidence` | `deterministic_reference` | 1.0000 | 1.980 | 0.0000 |
| `noisy_evidence` | `learned_constrained_reference` | 1.0000 | 1.900 | 0.0000 |
| `noisy_evidence` | `stale_certificate_control` | 0.9095 | 0.077 | 0.0932 |
| `noisy_evidence` | `stateless_reference` | 1.0000 | 1.800 | 0.0000 |
| `provenance_corrupt` | `certificate_amortized` | 1.0000 | 1.540 | 0.0000 |
| `provenance_corrupt` | `certificate_corrupted` | 1.0000 | 1.540 | 0.0000 |
| `provenance_corrupt` | `deterministic_reference` | 1.0000 | 1.480 | 0.0000 |
| `provenance_corrupt` | `learned_constrained_reference` | 1.0000 | 1.400 | 0.0000 |
| `provenance_corrupt` | `stale_certificate_control` | 0.9236 | 0.064 | 0.0788 |
| `provenance_corrupt` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 |
| `unresolvable` | `certificate_amortized` | 0.0000 | 0.240 | 0.0000 |
| `unresolvable` | `certificate_corrupted` | 0.0000 | 0.240 | 0.0000 |
| `unresolvable` | `deterministic_reference` | 0.0000 | 0.180 | 0.0000 |
| `unresolvable` | `learned_constrained_reference` | 0.0000 | 0.100 | 0.0000 |
| `unresolvable` | `stale_certificate_control` | 0.8870 | 0.026 | 0.0811 |
| `unresolvable` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |

## Results by held-out family

| Family | Policy | Accuracy | Mean cost | Harmful transfer |
| --- | --- | ---: | ---: | ---: |
| `interval` | `certificate_amortized` | 0.8798 | 1.067 | 0.0000 |
| `interval` | `certificate_corrupted` | 0.8798 | 1.114 | 0.0000 |
| `interval` | `deterministic_reference` | 0.8798 | 1.118 | 0.0000 |
| `interval` | `learned_constrained_reference` | 0.8798 | 1.038 | 0.0000 |
| `interval` | `stale_certificate_control` | 0.6326 | 0.064 | 0.3751 |
| `interval` | `stateless_reference` | 0.8798 | 1.205 | 0.0000 |
| `multiset` | `certificate_amortized` | 0.8734 | 1.095 | 0.0000 |
| `multiset` | `certificate_corrupted` | 0.8734 | 1.145 | 0.0000 |
| `multiset` | `deterministic_reference` | 0.8734 | 1.137 | 0.0000 |
| `multiset` | `learned_constrained_reference` | 0.8734 | 1.057 | 0.0000 |
| `multiset` | `stale_certificate_control` | 0.6126 | 0.071 | 0.3973 |
| `multiset` | `stateless_reference` | 0.8734 | 1.201 | 0.0000 |
| `tree` | `certificate_amortized` | 0.8633 | 1.091 | 0.0401 |
| `tree` | `certificate_corrupted` | 0.8633 | 1.100 | 0.0401 |
| `tree` | `deterministic_reference` | 0.8726 | 1.147 | 0.0000 |
| `tree` | `learned_constrained_reference` | 0.8726 | 1.071 | 0.0000 |
| `tree` | `stale_certificate_control` | 0.5996 | 0.064 | 0.4092 |
| `tree` | `stateless_reference` | 0.8726 | 1.227 | 0.0000 |
| `matrix` | `certificate_amortized` | 0.8789 | 1.147 | 0.0000 |
| `matrix` | `certificate_corrupted` | 0.8789 | 1.184 | 0.0000 |
| `matrix` | `deterministic_reference` | 0.8789 | 1.177 | 0.0000 |
| `matrix` | `learned_constrained_reference` | 0.8789 | 1.097 | 0.0000 |
| `matrix` | `stale_certificate_control` | 0.5993 | 0.064 | 0.4104 |
| `matrix` | `stateless_reference` | 0.8789 | 1.206 | 0.0000 |

## Results by memory size

| Memory size | Policy | Accuracy | Mean cost | Harmful transfer |
| ---: | --- | ---: | ---: | ---: |
| `10` | `certificate_amortized` | 0.8752 | 1.067 | 0.0016 |
| `10` | `certificate_corrupted` | 0.8752 | 1.106 | 0.0016 |
| `10` | `deterministic_reference` | 0.8756 | 1.120 | 0.0000 |
| `10` | `learned_constrained_reference` | 0.8756 | 1.040 | 0.0000 |
| `10` | `stale_certificate_control` | 0.6188 | 0.025 | 0.3821 |
| `10` | `stateless_reference` | 0.8756 | 1.207 | 0.0000 |
| `100` | `certificate_amortized` | 0.8648 | 1.062 | 0.0034 |
| `100` | `certificate_corrupted` | 0.8648 | 1.097 | 0.0034 |
| `100` | `deterministic_reference` | 0.8656 | 1.107 | 0.0000 |
| `100` | `learned_constrained_reference` | 0.8656 | 1.028 | 0.0000 |
| `100` | `stale_certificate_control` | 0.6321 | 0.043 | 0.3715 |
| `100` | `stateless_reference` | 0.8656 | 1.194 | 0.0000 |
| `1000` | `certificate_amortized` | 0.8868 | 1.129 | 0.0106 |
| `1000` | `certificate_corrupted` | 0.8868 | 1.164 | 0.0106 |
| `1000` | `deterministic_reference` | 0.8892 | 1.170 | 0.0000 |
| `1000` | `learned_constrained_reference` | 0.8892 | 1.091 | 0.0000 |
| `1000` | `stale_certificate_control` | 0.6015 | 0.067 | 0.4083 |
| `1000` | `stateless_reference` | 0.8892 | 1.230 | 0.0000 |
| `10000` | `certificate_amortized` | 0.8676 | 1.142 | 0.0294 |
| `10000` | `certificate_corrupted` | 0.8676 | 1.175 | 0.0294 |
| `10000` | `deterministic_reference` | 0.8736 | 1.182 | 0.0000 |
| `10000` | `learned_constrained_reference` | 0.8736 | 1.105 | 0.0000 |
| `10000` | `stale_certificate_control` | 0.5910 | 0.128 | 0.4333 |
| `10000` | `stateless_reference` | 0.8736 | 1.210 | 0.0000 |

## Gates

- `catalogue_completeness`: `True`.
- `test_separation`: `True`.
- `primary_safety`: `False`.
- `primary_efficiency_10_percent`: `False`.
- `invalidation_safety`: `False`.
- `corruption_detection`: `True`.
- `stale_control_exposed_failure`: `True`.
- `frozen_reference_exact`: `True`.

Failed gates remain valid results. These synthetic outcomes do not establish neural reasoning or CMS2 training value.

## Source hashes

- `experiments/crucible_c1j_cpu_stress_20261021/MODELS.json`: `27797d8094a33a7aa8b022e81058b514cd1fe67eae1d1e853e1255eab76c2ea4`
- `experiments/crucible_c1j_cpu_stress_20261021/RESULTS.json`: `d5be55d0d723b40026147f73f6c47d50568b0a0acb6684ba41c4121eb8752d2d`
- `experiments/crucible_c1j_cpu_stress_20261021/SPLIT_MANIFEST.json`: `bac5739b28d7b2d85d08c41f6dca7d829b255aa023dce679009f901a20b6e641`
- `experiments/crucible_c1k_constrained_procedure_20261021/RESULTS.json`: `e8d2cbdf24fbe7b083a7eefd6bfbb5e6c0afe282b51b0c090005634a79febd1f`
- `experiments/crucible_c1k_repeat_constrained_procedure_20261021/REPEAT_COMPARISON.json`: `6b42d392038a706dab7b3d497a64267a32c99f936b6b2c365425dcba5fdb89d4`
- `experiments/crucible_c1l_certificate_amortization_20261021/PROTOCOL.md`: `b65c9c688a8790fc0c6d7ae7ea3dc084a693a9167525835c5ae0d1580adbe80d`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
- `tools/run_crucible_c1k_constrained_procedure_20261021.py`: `1cbe59a4783127ee75d8d37568cbc38f3dcf376388411d20307059cd6bc494e2`
- `tools/run_crucible_c1l_certificate_amortization_20261021.py`: `dcb39d7c43fc05ee9dea0b569bde5e0036aded693c789544dedbd90290bef9dd`
