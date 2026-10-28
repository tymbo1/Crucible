# Crucible heterogeneous training-throughput benchmark

Classification: `engineering_benchmark_gpu_measured`

This is an engineering benchmark, not a scientific learning claim.

## Model

- Family: Crucible latent-attention + SwiGLU dense
- Size: small
- Parameters: 3,936,000

## Arm summary

| Arm | Runs | Mean tokens/s | Mean timed wall | GPU measured | CPU optimizer state |
|---|---:|---:|---:|---|---:|
| baseline_gpu | 2 | 33531.86 | 0.1916s | True | 0.00 MiB |
| ssre_v1_v3_delivery | 2 | 32943.27 | 0.1949s | True | 0.00 MiB |
| cpu_adamw_offload | 2 | 29448.99 | 0.2180s | True | 45.04 MiB |
| ssre_offload_verification | 2 | 29906.51 | 0.2148s | True | 45.04 MiB |

## Integrity

- Indexed input exact equivalence: `True`
- One-update CPU-offload equivalence: `True`
- One-update maximum parameter delta: `1.19e-07`
- Baseline/SSRE initial-state match: `True`
- Baseline/SSRE final-state match: `False`
- Selected device: `cuda`
- CUDA available: `True`

## Limitations

- The CPU-only result does not measure GPU throughput or PCIe transfer time.
- CPU AdamW offload is synchronized and intentionally not an asynchronous optimizer design.
- The dense benchmark model is a smaller representative Crucible-family model, not the future 500M model.
- The verification workers use deterministic source-record validation as a CPU workload proxy; they are not a full curriculum generator.
