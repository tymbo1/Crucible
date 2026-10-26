# Crucible Relevance and Adjacency Gate — Instrumentation Revision

This is the instrumentation-corrected continuation of the frozen-checkpoint
gate described in `CRUCIBLE_RELEVANCE_ADJACENCY_GATE_PROTOCOL_20251025.md`.

The v1 run is preserved as a failed instrumentation preflight because replacing
native SDPA with a mathematically equivalent hand implementation accumulated a
maximum logit difference above its `1e-5` tolerance. No behavioral result from
v1 is accepted as an attention result.

## Revision

The v2 forward path leaves native `scaled_dot_product_attention` responsible
for the model output. A wrapper captures Q, K, V and the mask, derives shadow
softmax weights for diagnostic summaries, and then calls the untouched native
SDPA operation for the returned output. The output comparison must therefore
be identical to native execution within the numerical comparison tolerance.

The attention values are explicitly labelled shadow probabilities: they are
derived from the inputs to native SDPA, not claimed to be native backend
telemetry.

All task conditions, checkpoints, variants, train/validation records and
sealed-test boundaries remain unchanged from v1.
