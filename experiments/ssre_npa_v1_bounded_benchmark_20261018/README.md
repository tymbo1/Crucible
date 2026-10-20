# SSRE-NPA v1 bounded-resource benchmark

This is a new versioned benchmark. The v0 artifact remains unchanged.

It is CPU-only and does not import Crucible, Braidformer, PyTorch, CUDA, RunPod, or credentials. It exercises the scheduling contract with deterministic synthetic work.

## What v1 adds

- a global in-flight packet window covering input queues, completed packets, reorder entries, staged packets, and unconsumed packets;
- condition-variable blocking with measured wait counts and wait durations;
- variable preparation costs and packet sizes;
- straggler workers and slow consumers;
- serial, fixed-worker, and adaptive deterministic modes;
- transient worker-failure injection with retry;
- cancellation, queue shutdown, and recovery checks;
- end-to-end timing, packet latency p50/p95, process CPU time, queue high-water marks, reorder memory, and payload-window bounds.

## Run

```bash
cd /home/timbushnell/ssre_npa_v1_bounded_benchmark_20261018
python3 bounded_benchmark.py --packets 192 --workers 4 --window 16 --queue-capacity 4 --output benchmark_results.json
```

This is a scheduler characterization benchmark, not a neural-training result. The benchmark explicitly reports that it used no GPU and no model training.
