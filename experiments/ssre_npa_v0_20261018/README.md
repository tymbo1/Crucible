# SSRE-NPA v0 artifact

This is an isolated, CPU-only design and mechanical validation package for a future neural pipeline accelerator.

It does **not**:

- modify Crucible or Braidformer code;
- attach to or interrupt the active context experiment;
- train a model or a controller;
- use CUDA, RunPod, credentials, or network services;
- claim a GPU-training speedup.

## Files

- `ARCHITECTURE.md` — system design, invariants, data path, controllers, telemetry, and isolation boundary.
- `INTERFACES.md` — packet, result, configuration, telemetry, and decision contracts.
- `PROFILING_PLAN.md` — staged profiling gates and adoption criteria.
- `interfaces.py` — transport-agnostic Python data contracts.
- `synthetic_benchmark.py` — serial versus bounded CPU reservoir benchmark.

## Run the mechanical benchmark

```bash
cd /home/timbushnell/ssre_npa_v0_20261018
python3 synthetic_benchmark.py --packets 256 --workers 4 --queue-capacity 8 --work-scale 4 --consume-delay-s 0.0002
```

The benchmark intentionally slows the consumer slightly so bounded-queue backpressure is exercised. It verifies that worker completion order does not alter the ordered result digest.

The benchmark output is disposable validation telemetry, not a Crucible experiment artifact.
