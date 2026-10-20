# SSRE-NPA v4 — local GPU step benchmark results

Status: **passed as a disposable engineering benchmark**.

This benchmark used fresh, disposable model/optimizer instances loaded from
the frozen context-128 E16 Branch-B seed-42 anchor. It did not modify the
frozen context package, any checkpoint, any manifest, or the active RunPod
experiment.

## Frozen inputs

- Device: NVIDIA GeForce RTX 3060 Ti
- Torch: 2.12.0+cu130; CUDA runtime 13.0
- Model: 170,233,600 parameters; E16; top-2; context 128; Branch B
- Workload: first 2 warm-up and 8 timed optimizer steps of the frozen seed-42
  schedule
- Repetitions: 2 per arm and telemetry mode
- Arms: current Python clone/stack assembly; indexed extraction from the
  original resident CPU tensor
- Manifest/config SHA-256:
  `15ed67b3e38db7a82ea6dcf7825f0e51c6ff2c66bd31da9aeead67aa3a309a72`
- Seed-42 anchor SHA-256:
  `317d2cc82f14f7877a3e84a76cf0f5bedaa0b1177389699f5246f6e998c3474e`
- Source JSONL SHA-256:
  `dca93acdadec204409f7312b6b5b6ea4897bdd846bbd4eb9a62c71b8dce36cce`
- Binary tensor SHA-256:
  `c8d6056d01f91ad391dba99a1b04d48e2b300829c5e6801d910249043bbebf88`
- Schedule SHA-256:
  `ec449a68230663f06d57589aa744626d596c5efaf1ba43c1b57fc5bb0cb32879`

## Equivalence gate

The deterministic control passed all four baseline/indexed comparisons:

- exact CPU input-prefix equivalence: **passed**;
- all losses and gradient norms finite: **passed**;
- maximum paired loss difference: **0.0**;
- final model-state hashes: **identical in every comparison**;
- all declared `1e-5` loss tolerances: **passed**.

The first non-deterministic diagnostic run is retained separately in
`results/`. Its approximately `4e-4`–`6e-4` loss drift was caused by CUDA
numerical nondeterminism, not by a batch mismatch. The deterministic result is
the equivalence result used here.

Deterministic enforcement used PyTorch's `warn_only` mode because the frozen
telemetry implementation uses CUDA `bincount`, for which this PyTorch build
has no deterministic implementation. That operation is aggregate-only and
outside the gradient path; its warning does not alter the model state. The
telemetry-off and telemetry-on model-state comparisons both passed exactly.

## Mean steady-state timing

Times below are totals across the 8 timed optimizer steps. The two repetitions
are averaged.

| Arm | Telemetry | Wall | CPU prep | H2D | Forward CUDA | Backward CUDA | Optimizer/clip | Non-compute proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current Python stack | off | 15.535 s | 11.017 ms | 5.725 ms | 4.675 s | 7.270 s | 260.2 ms | 296.1 ms |
| Indexed extraction | off | 15.308 s | 8.385 ms | 5.276 ms | 4.636 s | 7.207 s | 258.4 ms | 289.6 ms |
| Current Python stack | on | 17.133 s | 10.399 ms | 5.169 ms | 6.085 s | 7.231 s | 256.0 ms | 289.7 ms |
| Indexed extraction | on | 17.192 s | 8.654 ms | 5.144 ms | 6.089 s | 7.244 s | 258.1 ms | 289.7 ms |

Peak allocated VRAM was approximately **3.187 GiB** and peak reserved VRAM
approximately **3.971 GiB** in every mode.

## Findings

1. Indexed extraction reduced CPU batch-preparation time by approximately
   **23.9%** in the telemetry-off comparison, consistent with the v3 CPU
   replay. The absolute saving was only about **2.63 ms over eight steps**,
   or about **0.33 ms/step**.
2. End-to-end wall time improved by approximately **1.46%** with telemetry
   off: 15.535 s to 15.308 s. With telemetry enabled, indexed delivery was
   statistically indistinguishable at this short sample size and was slightly
   slower in the mean: 17.133 s to 17.192 s.
3. The dominant cost was neural execution: roughly 4.6–4.7 s of forward CUDA
   time and 7.2–7.3 s of backward CUDA time per 8-step window. CPU preparation,
   H2D transfer and the host/CUDA timing gaps were small by comparison.
4. Telemetry itself was expensive in this path. It increased wall time by
   about **1.60 s** for the current arm and **1.88 s** for indexed delivery
   over eight steps, primarily through an approximately 1.4–1.45 s increase
   in forward time. This is a stronger optimization target than a learned
   batch scheduler for this workload.
5. The forward host-versus-CUDA-event gap was approximately 1.5 ms over eight
   steps, and the corresponding backward gap approximately 1.2 ms. These
   measurements do not support CUDA launch/dispatch overhead as the dominant
   bottleneck in this benchmark. They are proxies, not hardware counters.

## Decision

Do **not** train SSRE-NPA's neural scheduler yet. The measured opportunity in
batch delivery is real but too small to justify a learned controller, and the
two-worker reservoir was already slower in v3. First investigate reducing or
decoupling telemetry overhead, then use a CUDA profiler if a precise MoE
dispatch attribution is required. Pinned/nonblocking staging and a larger
steady-state benchmark remain separate engineering tests; they were not
claimed here.

## Generated artifacts

- `gpu_step_benchmark.py`
- `MEASUREMENT_PROTOCOL.md`
- `README.md`
- `results/gpu_step_benchmark_summary.json` — initial non-deterministic
  diagnostic, retained unchanged
- `results_deterministic_v2/gpu_step_benchmark_summary.json` — accepted
  deterministic control
- `artifact_hashes.json` — generated after this report and all outputs were
  finalized

The final SHA-256 inventory is recorded in `artifact_hashes.json`.
