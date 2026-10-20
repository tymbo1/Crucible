# SSRE-NPA v4 — local GPU step benchmark

This is a short, disposable engineering benchmark for the frozen Crucible E16
Branch-B architecture. It loads fresh model instances from the immutable
seed-42 anchor and runs the same prefix of the frozen context-128 schedule
under two CPU batch-delivery arms:

- `current_python_stack`: per-sample clone plus `torch.stack`;
- `indexed_precomputed`: indexed extraction from the original resident CPU
  tensor, without creating a second dataset copy.

Each arm is run with telemetry disabled and enabled. The benchmark records
synchronized CPU/H2D/forward/backward/optimizer timings, CUDA-event timings,
VRAM peaks, process CPU time, finite-loss/gradient checks and final model-state
hashes. It does not save a checkpoint, modify a manifest, write to a frozen
experiment directory, or contact RunPod.

## Run

```text
python3 gpu_step_benchmark.py \
  --context-root /home/timbushnell/crucible_context_gate_v1_20261018/context_128 \
  --output-dir /home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018/results
```

Defaults are 2 warmup steps, 8 timed steps and 2 repetitions for seed 42.
The benchmark requires CUDA and BF16 support; if either is unavailable it
fails before loading a model.

## Interpretation boundary

The difference between synchronized host wall time and CUDA-event time is a
launch/synchronization proxy, not a direct hardware counter. The reported
non-compute interval is a GPU-idle proxy, not sampled GPU utilization. A later
CUDA profiler trace is required before changing the training runner or adding a
neural scheduler.

