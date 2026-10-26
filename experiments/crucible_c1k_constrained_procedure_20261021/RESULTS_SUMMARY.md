# C1k constrained learned procedure execution

Date: 2026-10-21
Execution: CPU-only replay; no new learner training, GPU, cloud or Crucible neural training.

## Aggregate paired results

| Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | Contract pass rate | Procedure failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deterministic` | 0.8761 | 1.0000 | 1.145 | 0.0000 | 0.2307 | 0 |
| `hybrid_unconstrained` | 0.8737 | 0.9973 | 1.002 | 0.0104 | 0.2307 | 0 |
| `learned_constrained` | 0.8761 | 1.0000 | 1.066 | 0.0000 | 0.2307 | 72 |
| `learned_unconstrained` | 0.8700 | 0.9930 | 0.994 | 0.0259 | 0.2307 | 0 |
| `oracle` | 1.0000 | 1.0000 | 0.000 | 0.0000 | 0.0000 | 0 |
| `persistent` | 0.5815 | 0.5383 | 0.000 | 0.4185 | 0.0000 | 0 |
| `stateless` | 0.8761 | 1.0000 | 1.210 | 0.0000 | 0.0000 | 0 |

## C1j failure attribution

- Frozen learned harmful-transfer events: `183`.
- Replayed learned harmful-transfer events: `183`.
- Exact event replay: `True`.

### Failures by condition
- `adaptation`: `74`
- `noisy_evidence`: `109`

### Failures by family
- `interval`: `27`
- `matrix`: `33`
- `multiset`: `22`
- `tree`: `101`

### Failures by memory_size
- `10`: `20`
- `100`: `24`
- `1000`: `62`
- `10000`: `77`

### Failures by decision
- `adapt`: `74`
- `reuse`: `109`

## Constrained-policy results by memory size

| Memory size | Accuracy | Mean cost | Harmful transfer | Contract rejections |
| ---: | ---: | ---: | ---: | ---: |
| `10` | 0.8756 | 1.040 | 0.0000 | 28 |
| `100` | 0.8656 | 1.028 | 0.0000 | 26 |
| `1000` | 0.8892 | 1.091 | 0.0000 | 44 |
| `10000` | 0.8736 | 1.105 | 0.0000 | 32 |

## Gates

- C1j failure-replay gate: `True`.
- Constrained accuracy gate: `True`.
- Constrained zero-harm gate: `True`.
- Constrained efficiency gate: `False`.
- Contract enforcement gate: `True`.

A failed gate remains a valid result. Passing synthetic gates does not establish neural reasoning or CMS2 training value.

## Source hashes

- `experiments/crucible_c1j_cpu_stress_20261021/MODELS.json`: `27797d8094a33a7aa8b022e81058b514cd1fe67eae1d1e853e1255eab76c2ea4`
- `experiments/crucible_c1j_cpu_stress_20261021/RESULTS.json`: `d5be55d0d723b40026147f73f6c47d50568b0a0acb6684ba41c4121eb8752d2d`
- `experiments/crucible_c1j_cpu_stress_20261021/SPLIT_MANIFEST.json`: `bac5739b28d7b2d85d08c41f6dca7d829b255aa023dce679009f901a20b6e641`
- `experiments/crucible_c1k_constrained_procedure_20261021/PROTOCOL.md`: `d3503d7e2ef8f1c026c07a2b1774113e0be9797e412099ea84dd364e6c4ff791`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
- `tools/run_crucible_c1k_constrained_procedure_20261021.py`: `1cbe59a4783127ee75d8d37568cbc38f3dcf376388411d20307059cd6bc494e2`
