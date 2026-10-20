# Crucible cross-GPU validation — RTX 4090

Date: 2026-10-20

This is a short engineering validation of the deterministic grouped-expert
engine. It is not a full scientific training run and does not replace the
frozen Crucible results.

## Protocol

- Device: NVIDIA GeForce RTX 4090
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Model: frozen E16 Branch B, 16 experts, top-2 routing, context 128
- Precision: BF16
- Seeds: 42 and 1337
- Repetitions: 2 per arm and seed
- Warmup: 2 optimizer steps
- Timed window: 8 optimizer steps
- Arms: original and rank-preserving grouped execution
- Telemetry: production telemetry on and diagnostic telemetry off
- Determinism: strict PyTorch mode with `CUBLAS_WORKSPACE_CONFIG=:4096:8`

The benchmark used the same manifest, input prefix, initial anchor and
optimizer contract for the paired arms within each seed. It was run in a
separate disposable RunPod pod; frozen manifests, checkpoints and prior
results were not modified.

## Production telemetry-on result

Wall times are the timed eight-step windows, averaged over two repetitions.

| Seed | Original (s) | Grouped (s) | Reduction | Throughput |
|---:|---:|---:|---:|---:|
| 42 | 25.254401 | 19.159639 | 24.1335% | 1.3181x |
| 1337 | 25.626717 | 19.264643 | 24.8259% | 1.3302x |
| Mean | 25.440559 | 19.212141 | **24.480%** | **1.324x** |

The 4090 therefore retains a substantial grouped-execution advantage, but the
measured reduction is smaller than the 32.02% production-telemetry result on
the RTX 3060 Ti. The optimization is hardware- and workload-dependent.

## Diagnostic telemetry-off result

| Seed | Original (s) | Grouped (s) | Reduction | Throughput |
|---:|---:|---:|---:|---:|
| 42 | 23.010794 | 16.684452 | 27.4929% | 1.3792x |
| 1337 | 23.369002 | 16.968199 | 27.3901% | 1.3772x |
| Mean | 23.189898 | 16.826325 | **27.442%** | **1.379x** |

## Timed CUDA phase averages, telemetry on

These are reported profiler scopes from the paired benchmark and are not
independent additive costs.

| Phase | Original (s) | Grouped (s) | Reduction |
|---|---:|---:|---:|
| Forward CUDA events | 10.776343 | 8.838856 | 17.977% |
| Backward CUDA events | 14.419598 | 10.130642 | 29.741% |
| Optimizer | 0.178506 | 0.178786 | -0.159% |
| CPU batch preparation | 0.017838 | 0.016381 | 8.070% |
| Host-to-device handoff | 0.006790 | 0.006585 | 3.048% |

The measured benefit is concentrated in forward and especially backward CUDA
execution. It is not explained by CPU batch preparation or optimizer work.

## Integrity outcome

- Seed 42: 8/8 runs finite; input-prefix equivalence passed; all paired final
  model-state hashes matched; loss tolerances passed.
- Seed 1337: 8/8 runs finite; input-prefix equivalence passed; all paired final
  model-state hashes matched; loss tolerances passed.
- The 1-warmup/2-timed-step seed-42 preflight also passed.
- No full-length 3,125-step original-versus-grouped equivalence was run on the
  4090. This remains a short cross-GPU engineering validation.

## Interpretation

The deterministic grouped engine transfers to the RTX 4090 with a reproducible
24.48% production-telemetry wall-time reduction across two seeds. This is
stronger than the original single-GPU observation, but it is not a universal
GPU speedup claim. The next safe use is as an optional, separately validated
execution backend; any longer scientific run still needs its own sustained
equivalence gate on the target GPU.

