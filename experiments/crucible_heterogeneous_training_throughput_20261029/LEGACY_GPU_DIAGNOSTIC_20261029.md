# Legacy GPU benchmark diagnostic — 2026-10-29

Source: commit `531c131`.

## Worker accounting

The legacy SSRE delivery arm declared four preparation threads. The combined
verification arm additionally declared four process workers. With eight
microbatches per step, one warmup step, two timed steps and two repetitions,
the combined arm submitted 32 timed verification tasks and 48 tasks including
warmup. Child-worker CPU utilization was not recorded; the report only had
parent-process CPU time and per-task elapsed time.

The verification path was not an independent streaming pipeline: each
preparation thread submitted one verification future and immediately waited on
`future.result()`. This bounded verification concurrency by the preparation
threads and prevented a clean measurement of independent CPU/GPU overlap.

## State mismatch

All 33 batches in the controlled comparison had exact direct/indexed input
equivalence. With default CUDA algorithms, baseline and SSRE final state hashes
differed slightly and the loss traces differed in the last floating-point
digits. Enabling deterministic CUDA algorithms made the loss traces identical
and restored exact final-state equality.

The mismatch was therefore CUDA floating-point kernel nondeterminism under the
tested FP16 configuration, not an input-order or indexed-assembly mismatch.

The replacement overlap benchmark enables deterministic CUDA algorithms and
measures CPU procedural work separately from the GPU training path.
