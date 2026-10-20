# Crucible grouped execution: staged sustained-equivalence plan

**Status:** Prepared only; not executed.

**Target:** Frozen E16 Branch B, rank-preserving grouped backend, production
telemetry enabled, RTX 4090.

**Authority:** This document does not authorize a GPU run, cloud rental, or
scientific training. It specifies the gate to review before execution.

## Objective

Determine whether the grouped backend preserves the complete training state,
not merely the model parameter state, for the frozen E16 experiment over at
least 3,125 optimizer steps.

The original implementation is the reference arm. The grouped backend must
use the same anchor, sample schedule, optimizer contract, numerical settings,
telemetry contract and checkpoint cadence.

## Required runner changes before launch

Create a new versioned runner; do not edit the sealed 4090 package in place.
The runner must add canonical, device-independent hashes for:

1. **Model state:** sorted parameter/buffer names, dtype, shape and contiguous
   CPU bytes.
2. **Optimizer state:** parameter-name mapping, parameter groups, scalar
   hyperparameters and every state tensor/scalar in canonical sorted order.
3. **Scheduler state:** canonical scalar/array serialization if a scheduler is
   present.
4. **RNG state:** Python `random`, NumPy, CPU Torch, every CUDA generator and
   any explicit generator objects.
5. **Sampler state:** schedule hash, committed step, batch index, batch
   boundaries and the exact next sample cursor.
6. **Training observables:** loss, gradient norm, valid/raw target counts,
   routing counts, auxiliary loss and finite-status flags.

Do not use pickle or device-specific serialization as the canonical digest.
Hash a typed record containing field name, dtype, shape and bytes; normalize
tensor values to CPU without changing the values before hashing. Save a
human-readable manifest next to every checkpoint.

## Staged gates

### Stage 0 — local static gate

Already completed for this assessment:

- Verify the sealed 4090 package index.
- Verify the manifest, anchors and schedule hashes.
- Compile the copied source files.
- Confirm no GPU or cloud process was started.

### Stage 1 — ten-step smoke gate

One seed, original and grouped, production telemetry on. Use separate fresh
processes and output directories. Compare state and RNG manifests after every
optimizer step. This catches checkpoint/hash plumbing errors cheaply.

**Pass:** exact model, optimizer, scheduler, RNG and sampler hashes at every
step; finite values; exact input fingerprints; no checkpoint corruption.

### Stage 2 — 100-step paired gate

Run seeds 42 and 1337. Capture checkpoints at steps 0, 1, 10, 25, 50 and 100.
Run the reference and grouped arms independently, then compare manifests
offline. Record per-step loss and gradient norms, plus all routing and
auxiliary-loss telemetry.

**Pass:** both seeds satisfy Stage 1's exact-state criteria at every checkpoint.
If any first divergence occurs, stop and preserve both arms for diagnosis.

### Stage 3 — full 3,125-step gate

Only after Stage 2 passes. Run both arms for both seeds with checkpoints at
0, 1, 10, 25, 50, 100, 250, 500, 750, 1,000, 1,500, 2,000, 2,500 and 3,125,
matching the frozen schedule's checkpoint policy where applicable.

Record:

- exact canonical model/optimizer/scheduler/RNG/sampler hashes;
- checkpoint and resume hashes;
- per-step or per-checkpoint loss and gradient-norm differences;
- routing/auxiliary-loss metrics;
- non-finite status;
- peak allocated/reserved VRAM;
- timed throughput and total wall time.

**Pass:** exact complete-state equality at every checkpoint for both seeds,
finite execution, resumability, and no integrity failures. A speed result is
reported separately and cannot override an equivalence failure.

## Predeclared decisions

- Any exact state-hash mismatch is a **failure**, not something to average away.
- If exact hashes fail but tensor-wise tolerances remain small, classify the
  result as numerical divergence and do not promote the backend; run a separate
  causal investigation.
- A grouped run that merely completes successfully is not a paired-equivalence
  pass.
- A loss-only match is insufficient.
- Cross-GPU runs may require a separate tolerance contract; this plan is for
  paired arms on the same target GPU and software stack.

Suggested secondary diagnostic tolerances, not substitutes for the exact gate:

- loss and gradient norm: absolute difference `1e-10`;
- tensor comparison: report max absolute and relative error per named tensor;
- routing counts: exact integer equality;
- weighted telemetry: exact equality where CPU float64 accumulation is part of
  the frozen contract, otherwise report the declared numeric tolerance.

## Proposed commands (not executed)

The current portable benchmark has no 3,125-step complete-state runner, so the
following are the commands to use after the new versioned runner is created.
They are intentionally shown as a specification, not as an executable claim.

```bash
cd /home/timbushnell
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 scripts/run_sustained_equivalence.py \
  --context-root /path/to/crucible_grouped_sustained_v1_20261020/context_128 \
  --seed 42 \
  --steps 10 \
  --checkpoint-interval 1 \
  --reference-arm original \
  --candidate-arm grouped_rank_preserving \
  --telemetry production \
  --output-dir /path/to/results/stage1_seed42
```

Repeat Stage 1 for seed 1337, then change `--steps` and checkpoint policy for
100 and 3,125 step stages. The runner should execute each arm in a fresh
process, write checkpoints atomically, and compare them in a separate
CPU-only verification pass.

Before any paid run, the exact resolved command, source hash, manifest hash,
anchor hashes and output paths must be recorded in a new immutable manifest.

## Hardware, runtime and cost estimate

The measured 4090 peak was approximately 3.233 GiB allocated and 4.0 GiB
reserved, so the tested model fits comfortably in a 24 GiB RTX 4090 under the
same context-128 settings. This is not a guarantee for longer context or
larger batch settings.

Using the sealed 4090 eight-step production-telemetry timings as a linear
planning estimate:

- original: `25.440559 / 8 * 3125` ≈ 2.76 hours per seed;
- grouped: `19.212141 / 8 * 3125` ≈ 2.08 hours per seed;
- two arms × two seeds: ≈ 9.7 GPU-hours before checkpoint and setup overhead.

This is only a planning estimate; the 100-step stage must recalibrate it.
Using the previously observed 4090 pod rate of **US$0.89/hour**, a conservative
working estimate is roughly US$10–12 including 20–30% overhead. Set a hard
maximum proposed spend of **US$15** for one 4090 sustained gate, and stop before
that cap if the staged gate fails. Verify the live price before authorization;
the price is not a frozen scientific constant.

No cloud rental is authorized by this plan.

## Artifact and shutdown protocol

Use a persistent volume or a resumable upload path. After every stage and every
full-run checkpoint:

1. flush and close the checkpoint;
2. verify its canonical manifest and SHA-256;
3. copy it to persistent storage;
4. independently verify the copied hash;
5. retain the pod until the final result index is present;
6. only then stop/delete the pod and verify the billing state.

The final artifact directory must contain the source/config/manifest hashes,
per-arm logs, checkpoint manifests, comparison reports, timing summaries,
failure diagnostics and a final hash index. Do not overwrite the 4090 package
or any previous Crucible analysis.

## Stop conditions

Stop immediately on:

- any exact-state mismatch;
- NaN/Inf, CUDA/device failure or checkpoint corruption;
- schedule, sample-order or provenance mismatch;
- missing or duplicate checkpoint;
- unexplained source/config hash change;
- projected spend reaching US$15;
- runtime or memory behaviour exceeding the preflight estimate materially.

## Decision after the gate

- **Pass:** retain grouped execution as a validated opt-in backend for the
  tested configuration and document the measured speedup separately.
- **Fail with numerical divergence:** keep the original as scientific
  reference, preserve the grouped result as an engineering artifact, and
  investigate the first divergence before any default adoption.
- **Fail mechanically:** repair the runner or environment; do not infer a
  model/architecture result.
