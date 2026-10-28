# Crucible heterogeneous training-throughput benchmark

This is an engineering benchmark for the local Ryzen 7 5700X / RTX 3060 Ti
system. It is not a scientific Crucible experiment and does not modify frozen
research artifacts.

The benchmark reuses the frozen context-128 byte corpus and schedule, but
trains a fresh smaller dense model built from Crucible's latent-attention and
SwiGLU primitives. It compares:

1. direct batch assembly with ordinary AdamW;
2. SSRE-NPA v1-style bounded scheduling plus SSRE-NPA v3 indexed delivery;
3. synchronized CPU-master AdamW offload;
4. indexed delivery, four process verification workers and CPU-master AdamW.

The v1 bounded-window contract provides backpressure and ordered commits. The
v3 indexed path avoids repeated per-sample cloning. The optimizer remains a
single synchronized owner; SSRE workers never mutate parameters or optimizer
state.

## Run

CUDA is optional. With the current environment, the expected CPU validation
command is:

```bash
python3 experiments/crucible_heterogeneous_training_throughput_20261029/benchmark.py \
  --device auto \
  --model-size small \
  --warmup-steps 1 \
  --timed-steps 2 \
  --repetitions 2 \
  --window 2 \
  --workers 4 \
  --output-dir experiments/crucible_heterogeneous_training_throughput_20261029/results_cpu
```

If CUDA is available, the same command selects CUDA automatically and records
CUDA-event forward/backward timing, host-to-device bytes, peak VRAM and a GPU
idle-time proxy. On the current machine `nvidia-smi` cannot communicate with
the driver, so GPU results must be reported as unmeasured until the driver is
restored.

## Interpretation boundary

The result measures pipeline behavior, not 500M-model training time. The
benchmark model is deliberately smaller so the four arms can be compared
quickly. CPU-only results validate implementation and equivalence paths but
cannot establish GPU throughput or PCIe overlap.
