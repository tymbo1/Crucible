# Concurrent CPU procedural-work / GPU training benchmark

Classification: `exploratory_gpu_overlap_measured`

The CPU work is Crucible's actual frozen-grammar candidate generation and independent verifier path. It is attached to future training microbatches and consumed in order; it does not affect model gradients.

## Results

| Condition | End-to-end wall (s) | Neural tokens/s | Verified procedures/s | Workers | CPU worker utilization | Max queue | GPU idle proxy (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpu_only | 2.809 | 22806.5 | 0.0 | 0.0 | 0.0% | 0 | 0.151 |
| sequential_cpu | 3.024 | 21065.1 | 31.4 | 1.0 | 11.8% | 0 | 0.149 |
| concurrent_ssre | 2.882 | 22101.5 | 33.0 | 4.0 | 3.9% | 8 | 0.152 |

Sequential/concurrent wall-time speedup: **1.050x**.

## Controls

- Same task manifest: `7c47ea3b6fbbe8f8478fa2f6b90b2dd12ce22af18976c132295394a97f12f060`
- All-batch input equivalence: `True`
- Deterministic CUDA algorithms: `True`
- Sequential/concurrent task-result digest match: `True`
- GPU-only versus concurrent final-state match: `True`
- Repetitions per condition: `3`

## Interpretation boundary

This measures useful-work overlap, not neural learning quality. It is an engineering workload measurement, not a curriculum or scientific training result.
