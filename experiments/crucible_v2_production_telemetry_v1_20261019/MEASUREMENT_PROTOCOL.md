# Production telemetry validation protocol

## Fixed intervention

Only these two observational reductions differ from the frozen implementation:

- weighted selected-probability accumulation;
- weighted selected-count accumulation.

Both use CPU `torch.bincount` with float64 weights, then copy the resulting
expert-length vectors to the existing GPU telemetry accumulators. All other
telemetry and all model computations remain unchanged.

## Gates

1. Strict 100-step original-versus-grouped equivalence for seeds 42 and 1337.
2. Exact model, gradient and optimizer-state SHA-256 equality.
3. Exact four-layer telemetry equality within the existing tolerance.
4. Paired warmed overhead measurements with telemetry disabled and enabled.
5. Only after gates 1–4 pass, run the immutable 3,125-update schedule for the
   grouped engine.

## Interpretation

The 3,125-update run is a sustained validation of the corrected grouped
implementation, not a new training experiment or a replacement for the frozen
scientific results. No cross-GPU claim is made from this local run.

The short benchmark and sustained metrics must report timed execution,
validation loss, routing telemetry, nonfinite events and provenance hashes
separately. Any divergence stops interpretation and preserves the failure.
