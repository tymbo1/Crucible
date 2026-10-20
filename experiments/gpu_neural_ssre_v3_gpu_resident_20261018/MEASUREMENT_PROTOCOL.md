# GPU Neural SSRE v3 measurement protocol

## Scope

This study measures the cost and integrity of routing-metadata placement. It
does not train a scheduler and does not alter the scientific Crucible runs.

## Frozen inputs

- E16 Branch B layer-0 expert modules from the immutable context-128 anchor;
- top-2 routing-shaped synthetic assignments;
- identical input activations and routing weights for every variant;
- balanced and highly skewed fixtures;
- 512 routed tokens in the preflight.

## Candidate paths

### `cpu_offsets`

The V2 accepted rank-preserving grouped implementation. Expert/rank counts
and offsets are copied to host Python lists before the expert loop.

### `gpu_offsets_scalar`

Counts and prefix sums are constructed on CUDA. The existing Python loop reads
two scalar boundaries per expert/rank segment. Those reads are intentionally
recorded as a remaining synchronization boundary.

### `gpu_segment_mask`

Sorted pair IDs remain on CUDA. Each expert/rank segment is selected with a
CUDA boolean mask, so no host offset list or scalar boundary is required.
This preserves per-segment matrix shapes but may trade synchronization for
repeated device-side scans.

## Integrity contract

For every candidate:

- routing assignments and GPU/CPU metadata must agree;
- output must remain within the V2 forward tolerance of `5e-3`;
- expert gradients must remain within the V2 gradient tolerance of `5e-4`;
- losses must be finite;
- no model or optimizer state is updated;
- timed trials use the same input tensors and warm-up policy.

## Timing

CUDA events measure forward and backward device time. Host wall time includes
Python dispatch and synchronization. Warm-up trials are excluded. No
profiler is enabled during timing.

## Interpretation

The result can justify a later full-model experiment only if a candidate is
integrity-safe and provides a repeatable end-to-end improvement. A metadata
placement change that is slower, or that leaves the Python launch loop as the
dominant cost, is recorded as a negative feasibility result rather than being
used to motivate a neural scheduler.

