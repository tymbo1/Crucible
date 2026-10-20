# Measurement protocol

## Reference and intervention

The frozen context runner is the scientific reference. The intervention is
V2's deterministic `grouped_rank_preserving` MoE implementation, applied in
memory to every MoE layer after model construction and before loading the
immutable seed anchor.

The intervention changes execution organization only. It preserves the router,
top-k decisions, routing weights, expert parameters, loss, optimizer and sample
schedule. It is not a new model or a new training condition.

## Integrity requirements

- Verify the manifest SHA before running.
- Verify the model-core provenance SHA through the frozen runner.
- Reuse the frozen anchor and schedule SHA checks.
- Record the wrapper, V1, V2, model-core and context-runner hashes.
- Write metrics to this new versioned directory only.
- Require finite recorded losses and complete expected checkpoint rows.
- Preserve the original RunPod context-128 metrics as the external baseline;
  do not overwrite them.

## Timing and interpretation

The run reports the frozen runner's interval wall time, CUDA forward/backward
timings, validation metrics, routing telemetry and memory. A sustained result
must not be inferred from the earlier short V2 benchmark alone.

Any numerical divergence from the original execution is reported. No claim of
bitwise identity is made unless the recorded hashes and declared tolerances
support it.
