# SSRE-NPA v5 — telemetry-efficiency experiment

This is an isolated, disposable GPU benchmark for the frozen Crucible E16
Branch-B path. It imports the frozen model implementation read-only and does
not modify checkpoints, manifests, training runs, or RunPod jobs.

The benchmark compares:

- `baseline_off`: no telemetry;
- `current_full`: the frozen `ResearchMoETelemetry` implementation;
- `gpu_full_batched`: all frozen full-diagnostic fields with vectorized GPU
  accumulation and one packed host transfer at summary time;
- `minimal_scientific`: the primary byte-affinity/load fields needed by the
  preregistered evaluator, omitting optional per-token selection and
  expert-output diagnostic matrices;
- `buffered_full`: full optimized accumulation with deferred report
  serialization.

Each backend uses fresh model/optimizer instances, the same seed-42 frozen
schedule prefix, two warm-up steps, eight timed steps and two repetitions.

The accepted output is:

`full_run_v3/telemetry_efficiency_summary.json`

The one-step mechanical preflight is retained in `preflight_v3/`.

