# SSRE-NPA v3 — measured CPU batch-delivery replay

This is an isolated, read-only measurement artifact. It uses the frozen local
Crucible context-128 package and does not modify any source, manifest,
checkpoint, metric file or training run. It does not import the Crucible model,
call CUDA, rent cloud hardware or start training.

## What is measured

The replay measures the CPU-side path that feeds the current runner:

- source JSONL UTF-8 validation and byte-length accounting, separately;
- exact source-to-binary token reconstruction for all training examples;
- schedule loading and permutation validation for seeds 42 and 1337;
- current Python per-sample clone + `torch.stack` batch assembly;
- an indexed precomputed CPU tensor representation;
- an ordered, bounded CPU worker reservoir over the indexed representation.

Every batch records x/y digests, PAD/non-PAD and raw-byte mask digests, ordered
index digests, source-byte counts, target counts and batch boundaries. Candidate
arms are compared against the current path; a speed result is rejected if any
digest or count differs.

## What is not measured

There is no GPU forward/backward pass, host-to-device copy, CUDA event, kernel
dispatch trace or MoE execution in this gate. The local NVIDIA driver was not
communicating when the gate began, and the active RunPod experiment is kept
separate. The results therefore identify CPU delivery cost only and cannot
attribute total training time to GPU starvation.

## Run

```text
python3 cpu_replay.py \
  --context-root /home/timbushnell/crucible_context_gate_v1_20261018/context_128 \
  --output-dir /home/timbushnell/ssre_npa_v3_measured_batch_replay_20261018/results
```

The default uses one warmup and two timed repetitions for both frozen schedules
and a two-worker, eight-batch bounded reservoir. The output is new and
versioned under this directory.

