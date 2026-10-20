# GPU Neural SSRE v1 — deterministic grouped-expert result

Date: 2026-10-18  
Device: NVIDIA GeForce RTX 3060 Ti  
Workload: frozen Crucible E16 Branch-B, context 128, seed 42  
Timing: two warm-up steps, eight timed optimizer steps, two repetitions

## Accepted equivalence-preserving arms

`grouped_rank_preserving` and `grouped_rank_largest_first` both passed the
one-step forward, gradient and optimizer equivalence probe with zero measured
delta. Their final full-run model-state hashes also exactly matched the
original arm in both repetitions.

| Arm | Mean timed wall | Forward CUDA | Backward CUDA | Tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Original frozen loop | 12.312 s | 4.712 s | 7.316 s | 2,072 | 3.189 GiB |
| Rank-preserving grouped | 7.737 s | 3.203 s | 4.235 s | 3,297 | 3.187 GiB |
| Rank-preserving largest-first | 7.739 s | 3.204 s | 4.235 s | 3,296 | 3.187 GiB |

Relative to the original, rank-preserving grouping reduced total timed wall
time by **37.2%**, increased measured throughput by approximately **59.1%**,
reduced forward CUDA time by **32.0%**, and reduced backward CUDA time by
**42.1%**. Optimizer time remained approximately 0.258 s in all arms.

Largest-first ordering was effectively neutral and slightly slower than the
default rank-preserving order. There is no evidence that a workload-order
policy is needed for this fixed workload.

## Exploratory non-equivalent arms

The faster arms below are retained as diagnostic evidence, not accepted
baselines:

| Arm | Wall reduction | Forward max abs delta | Loss max abs delta | Status |
|---|---:|---:|---:|---|
| One-call grouped | 57.3% | 0.17383 | 8.05e-5 | Rejected |
| Padded batched GEMM | 70.7% | 0.17969 | 1.46e-4 | Rejected |

Combining both routing ranks changes the expert GEMM shape and therefore the
BF16 kernel tiling. The resulting output drift is much larger than the
declared 5e-3 logit tolerance, despite relatively small gradient deltas.
Those arms cannot be used as scientific speed references without an explicit
numerical-contract decision.

## Integrity

- Frozen input prefix: exact.
- All timed runs finite: yes.
- Accepted grouped arms: exact one-step equivalence and exact final state
  hashes.
- Optimizer updates: disposable benchmark updates only.
- Frozen Crucible manifests, anchors and results: unchanged.
- Neural scheduler: not trained or evaluated.

## Controlled dispatch fixtures

Using the real frozen E16 expert weights, a dispatch-only control compared
exactly balanced assignments with a deliberately skewed 28.2:1 max:min
assignment ratio:

| Fixture | Original forward+backward | Rank-preserving grouped | Relative reduction |
|---|---:|---:|---:|
| Balanced, ratio 1.0 | 14.985 ms | 9.700 ms | 35.3% |
| Imbalanced, ratio 28.2 | 15.119 ms | 10.965 ms | 27.5% |

The padded batched arm was 37.0% faster on the balanced fixture but 114.2%
slower on the imbalanced fixture because padding every expert to the dominant
group size dominated the computation. It is therefore not a generally valid
strategy for sparse expert workloads.

## Decision

The deterministic GPU execution hypothesis has passed its first meaningful
gate. GPU-resident token grouping can substantially reduce execution time
without changing the model’s numerical trajectory, provided expert GEMM
shapes and rank accumulation order are preserved.

The next neural-scheduler experiment is not yet justified: the accepted
deterministic baseline already captures the large improvement. A future
controller would need to choose among this baseline and additional validated
plans on workloads with changing expert shapes or imbalance, then beat the
best deterministic policy including controller overhead.

