# Concurrent CPU procedural-work / GPU training benchmark

Classification: `exploratory_gpu_overlap_measured`

The CPU work is deterministic synthetic procedure generation and exhaustive verification. It is attached to future training microbatches and consumed in order; it does not affect model gradients.

## Results

| Condition | End-to-end wall (s) | Neural tokens/s | Verified procedures/s | Workers | CPU worker utilization | Max queue | GPU idle proxy (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpu_only | 2.784 | 23013.9 | 0.0 | 0.0 | 0.0% | 0 | 0.150 |
| sequential_cpu | 3.354 | 18989.5 | 50.1 | 1.0 | 19.0% | 0 | 0.150 |
| concurrent_ssre | 2.848 | 22360.9 | 59.0 | 4.0 | 6.6% | 8 | 0.148 |

Sequential/concurrent wall-time speedup: **1.177x**.

## Controls

- Same task manifest: `118b71ed3165eebf8fe58b982bd9e758bcb53797580c2b44508a6c36ff9b229a`
- All-batch input equivalence: `True`
- Deterministic CUDA algorithms: `True`
- Sequential/concurrent task-result digest match: `True`
- GPU-only versus concurrent final-state match: `True`
- Repetitions per condition: `3`

## Interpretation boundary

This measures useful-work overlap, not neural learning quality. The synthetic procedure generator is a CPU workload proxy for Crucible's future generation/verification work; it is not a curriculum or scientific training result.
