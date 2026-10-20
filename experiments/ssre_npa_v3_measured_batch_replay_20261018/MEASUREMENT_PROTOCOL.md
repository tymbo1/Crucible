# SSRE-NPA v3 measurement protocol

## Reference contract

The reference is the current E16 context runner's training assembly:

```text
for index in batch_indices:
    dataset[index]["x"].clone()
    dataset[index]["y"].clone()
torch.stack(x_items)
torch.stack(y_items)
```

The replay uses the frozen context-128 manifest, the actual `stories_train.bin`,
the actual packed train JSONL and both immutable dynamic source-byte schedules.
It preserves each schedule's optimizer-step grouping and microbatch boundaries.

## Arms

| Arm | Description |
|---|---|
| `current_python_stack` | Direct reproduction of per-sample slicing/cloning and stack assembly. |
| `indexed_precomputed` | One resident CPU `[example, x/y, position]` tensor and indexed batch extraction. |
| `cpu_reservoir` | Persistent CPU worker threads, bounded ordered future window, indexed extraction and ordered commit. |

The reservoir's worker completion may be out of order, but its committed batch
sequence is the frozen schedule order. The window is a real bound on pending
batch results and the payload size of every emitted batch is recorded.

## Integrity checks

- Manifest/config and source/binary/schedule SHA-256 checks.
- Exact UTF-8 source-to-token reconstruction for all 100,000 train examples.
- Exact BOS/EOS/PAD geometry and valid/raw target masks.
- Exact schedule permutation, step source bytes and target counts.
- Candidate x/y/mask/index fingerprints compared with the reference for every
  timed batch.

## Timing discipline

- CPU Torch threads are set to one to make the delivery comparison explicit.
- One warmup repetition is discarded.
- Two timed repetitions are retained per arm and seed.
- Assembly, worker preparation, queue wait, hashing and end-to-end wall time
  are recorded separately.
- Source validation and precomputed-storage setup are reported outside the
  steady-state batch-delivery timings.

## Interpretation boundary

An indexed or reservoir speedup is evidence about CPU batch delivery only. It
does not establish GPU starvation, CUDA overlap, or end-to-end training gain.
Those require a later authorized adapter with synchronized CUDA timing and
profiler traces, after the current RunPod experiment is sealed.

