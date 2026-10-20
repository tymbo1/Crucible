# SSRE-NPA v3 measured CPU batch-replay results

Date: 2026-10-18 (Brisbane)

## Boundary

This gate used the immutable local context-128 Crucible package only. It did
not import the Crucible model, run a forward/backward pass, call CUDA, modify a
manifest/checkpoint/result, contact RunPod, or interrupt the active context
experiment. There was no local Crucible training process at the start. GPU
timing is intentionally absent from this CPU-only gate.

Frozen configuration binding:

`experiment_config.json` / manifest SHA-256:

`15ed67b3e38db7a82ea6dcf7825f0e51c6ff2c66bd31da9aeead67aa3a309a72`

## Integrity

- 100,000 source records loaded and SHA-validated.
- 204,800,000-byte training binary loaded and SHA-validated.
- All 100,000 source records reconstructed exactly into binary x/y tokens.
- BOS, EOS, PAD, non-PAD and raw-byte target masks: 0 mismatches.
- Both frozen schedules validated as exact single permutations.
- Seeds: 42 and 1337.
- Timed repetitions: 2 per arm and seed, after 1 warmup repetition.
- Candidate-vs-reference batch fingerprint comparisons: 8/8 exact matches.
- Timed trace records: 312,192 batch events.

## Measured CPU delivery

Means across both seeds and both timed repetitions:

| Arm | Wall time/run | Batch/s | Source MB/s | Batch assembly | Queue wait |
|---|---:|---:|---:|---:|---:|
| Current Python stack | 2.627 s | 9,904 | 3.75 | 0.958 s | 0 |
| Indexed precomputed | 2.082 s | 12,495 | 4.73 | 0.449 s | 0 |
| CPU reservoir, 2 workers / window 8 | 3.496 s | 7,443 | 2.82 | 1.463 s worker prep | 181 ms |

Relative to the current assembly path, this particular indexed CPU layout
reduced end-to-end CPU replay time by about 20.7% and increased delivery
throughput by about 26.2%, with exact input equivalence.

The two-worker reservoir was about 33.1% slower than the current path in this
small-batch CPU-only configuration. It incurred approximately 181 ms of total
ordered-consumer wait per run. This does not reject a reservoir for a real GPU
consumer; it says that this configuration adds overhead when the consumer is
only a CPU replay and batches are small.

## Memory and preparation costs

- Source JSONL validation: approximately 852 ms.
- Binary load and hash: approximately 256 ms.
- Full source-to-binary token/mask verification: approximately 801 ms.
- Precomputed resident tensor construction: approximately 82 ms.
- Peak process RSS observed during the replay: approximately 1.01 GB.
- Precomputed tensor storage: 204.8 MB.

These setup costs are reported separately from steady-state batch delivery.

## Attribution

The measured result supports a concrete CPU-side optimization target:
precomputed/indexed batch extraction removes much of the current per-sample
clone/stack cost. It does not establish that the GPU was waiting for data.

The reservoir result is not a GPU result and cannot answer whether asynchronous
CPU preparation overlaps with Crucible CUDA work. The next measurement needed
for that question is an authorized live adapter that records synchronized CUDA
events around H2D, forward/backward, MoE dispatch/telemetry and optimizer
boundaries while preserving the same schedule.

## Recommendation

Do not train the neural scheduler yet. First use the indexed representation as
the CPU baseline and attach a measured, bounded reservoir to a disposable
short CUDA preflight only after the active RunPod experiment is sealed. Adopt a
learned scheduler only if it beats this deterministic indexed/reservoir
baseline after queue, worker and controller overhead.

