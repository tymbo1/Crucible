# Crucible heterogeneous training-throughput benchmark

This is an engineering benchmark for a local development system. It is not a
scientific Crucible experiment and does not modify frozen research artifacts.

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

CUDA is optional. Set `CRUCIBLE_CONTEXT_ROOT` to the local context-128 corpus
or pass `--context-root` explicitly. The benchmark redacts local paths and
host metadata by default; `--include-machine-metadata` is an explicit local
diagnostic opt-in.

The CPU validation command is:

```bash
export CRUCIBLE_CONTEXT_ROOT=/path/to/context_128
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
idle-time proxy. GPU availability and the exact blocker are recorded separately
in the GPU diagnostic artifact; no driver changes are performed by this
benchmark.

## Interpretation boundary

The result measures pipeline behavior, not 500M-model training time. The
benchmark model is deliberately smaller so the four arms can be compared
quickly. CPU-only results validate implementation and equivalence paths but
cannot establish GPU throughput or PCIe overlap.

## Independent CPU/GPU overlap benchmark

`concurrent_procedural_overlap_benchmark.py` compares a GPU-only reference,
synchronous CPU procedure generation/verification, and a bounded four-process
SSRE producer running concurrently with GPU training. It uses deterministic
CUDA algorithms, identical task manifests, ordered commits and sanitized
reports:

```bash
export CRUCIBLE_CONTEXT_ROOT=/path/to/context_128
python3 experiments/crucible_heterogeneous_training_throughput_20261029/concurrent_procedural_overlap_benchmark.py \
  --model-size small \
  --warmup-steps 3 \
  --timed-steps 20 \
  --workers 4 \
  --window 8 \
  --verification-rounds 128 \
  --repetitions 3 \
  --output-dir experiments/crucible_heterogeneous_training_throughput_20261029/results_overlap_gpu
```

The primary comparison is end-to-end wall time for the same verified
procedural work. The CPU procedure generator is an engineering workload proxy;
it is not a curriculum or scientific learning result.

To measure the actual Crucible candidate-generation and independent-verifier
path, use the prepared source/operator-validation manifests:

```bash
export CRUCIBLE_CONTEXT_ROOT=/path/to/context_128
python3 experiments/crucible_heterogeneous_training_throughput_20261029/concurrent_procedural_overlap_benchmark.py \
  --workload actual \
  --model-size small \
  --warmup-steps 3 \
  --timed-steps 20 \
  --workers 4 \
  --window 8 \
  --repetitions 3 \
  --output-dir experiments/crucible_heterogeneous_training_throughput_20261029/results_overlap_actual_gpu
```

This actual-workload path uses source and operator-validation AST metadata to
construct the operator state, but passes only demonstrations and independent
verification examples to CPU workers. It does not open sealed target answers.
