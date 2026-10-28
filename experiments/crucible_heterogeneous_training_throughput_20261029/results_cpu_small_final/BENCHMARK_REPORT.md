# Crucible heterogeneous training-throughput benchmark

Classification: `engineering_benchmark_cpu_validated_gpu_unmeasured`

This is an engineering benchmark, not a scientific learning claim.

## Model

- Family: Crucible latent-attention + SwiGLU dense
- Size: small
- Parameters: 3,936,000

## Arm summary

| Arm | Runs | Mean tokens/s | Mean timed wall | GPU measured | CPU optimizer state |
|---|---:|---:|---:|---|---:|
| baseline_gpu | 2 | 11179.09 | 0.5742s | False | 0.00 MiB |
| ssre_v1_v3_delivery | 2 | 11090.16 | 0.5788s | False | 0.00 MiB |
| cpu_adamw_offload | 2 | 11031.09 | 0.5819s | False | 45.04 MiB |
| ssre_offload_verification | 2 | 11051.24 | 0.5809s | False | 45.04 MiB |

## Integrity

- Indexed input exact equivalence: `True`
- One-update CPU-offload equivalence: `True`
- One-update maximum parameter delta: `0`
- Baseline/SSRE initial-state match: `True`
- Baseline/SSRE final-state match: `True`
- Selected device: `cpu`
- CUDA available: `False`

## Limitations

- The CPU-only result does not measure GPU throughput or PCIe transfer time.
- CPU AdamW offload is synchronized and intentionally not an asynchronous optimizer design.
- The dense benchmark model is a smaller representative Crucible-family model, not the future 500M model.
- The verification workers use deterministic source-record validation as a CPU workload proxy; they are not a full curriculum generator.
