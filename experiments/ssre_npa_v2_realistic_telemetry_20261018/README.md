# SSRE-NPA v2 — realistic scheduler telemetry

This is a separate, CPU-only engineering artifact for the next SSRE-NPA gate.
It does not modify, resume, benchmark, or otherwise interact with Crucible
training, the frozen context-gate artifacts, the v0/v1 SSRE-NPA artifacts, or
the active RunPod pods.

## Purpose

v1 established the execution contract and bounded-resource behavior. Its
synthetic benchmark was intentionally small and proved integrity, not useful
neural-pipeline speed. v2 adds two things before any controller is trained:

1. a read-only static audit of the actual E16 context runner and model routing
   path; and
2. a deterministic workload/telemetry framework covering variable preparation
   cost, payload size, cache state, stragglers, slow consumers and telemetry
   overhead.

The generated traces are **deterministic virtual workload proxies**, not GPU
benchmarks. They are suitable for validating schema, replay and feature
coverage. They must not be reported as measured training acceleration.

## Files

- `TELEMETRY_SCHEMA.md` — event schema, invariants and derived metrics.
- `WORKLOAD_MATRIX.json` — workload scenarios and scheduler arms.
- `runner_static_audit.py` — AST/source audit; it never imports or executes
  the Crucible runner.
- `trace_collector.py` — deterministic virtual trace generator and validator;
  it has no Torch or CUDA dependency.
- `runner_static_audit.json` — audit of the current E16 runner/model source.
- `telemetry_trace.jsonl` — generated event stream.
- `telemetry_summary.json` — aggregate trace metrics.
- `artifact_hashes.json` — SHA-256 record for this v2 package's artifacts.

## Commands

```text
python3 runner_static_audit.py --output runner_static_audit.json
python3 trace_collector.py --matrix WORKLOAD_MATRIX.json \
  --trace-output telemetry_trace.jsonl \
  --summary-output telemetry_summary.json
```

The collector uses virtual milliseconds and deterministic seeds. A future
live adapter may emit the same event schema from the real SSRE-NPA scheduler,
but must preserve packet order, include resource measurements, and label
`source_kind` as measured rather than synthetic.

## Decision boundary

No neural controller is trained here. The next legitimate gate is a real
trace-collection adapter around the deterministic scheduler, followed by an
offline train/validation split for controller prediction. A learned controller
must beat a competitive deterministic heuristic after controller overhead and
must never be allowed to alter packet order, model inputs, optimizer state or
scientific experiment semantics.

