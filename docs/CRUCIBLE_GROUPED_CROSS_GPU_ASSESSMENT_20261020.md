# Crucible grouped execution: RTX 4090 assessment

**Audit date:** 2026-10-20

**Scope:** Local inspection and CPU-only verification of the sealed package
`/home/timbushnell/crucible_cross_gpu_4090_v1_20261020/`.

**Execution boundary:** No GPU benchmark, cloud rental, pod launch, training run,
or modification of frozen Crucible results was performed for this audit.

## Executive verdict

**PASS WITH LIMITATIONS — cross-GPU short-run engineering validation.**

The rank-preserving grouped backend reproduced a substantial RTX 4090 speed
advantage across both seeds and both telemetry modes. All 14 files listed in
the package hash index match their recorded SHA-256 values, and the included
Python sources compile successfully.

The result is not yet a sustained training-equivalence result. The benchmark's
`final_state_sha256` covers the model `state_dict` only. It does not hash the
optimizer state, scheduler state, RNG state, sampler state, or telemetry
accumulators. The grouped backend therefore remains an opt-in engineering
backend until the staged sustained gate in
`docs/CRUCIBLE_GROUPED_SUSTAINED_EQUIVALENCE_PLAN_20261020.md` passes.

## Frozen experiment

The package records:

- NVIDIA GeForce RTX 4090
- PyTorch `2.8.0+cu128`; CUDA runtime `12.8`
- E16 Branch B, 170,233,600 trainable parameters
- 16 experts, top-2 routing, context length 128
- BF16 autocast
- Seeds 42 and 1337
- Original versus rank-preserving grouped execution
- Production telemetry on and diagnostic telemetry off
- Two warmup optimizer steps, eight timed optimizer steps, two repetitions
- Strict deterministic mode in the portable driver, with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` required by the contract

The manifest specifies four layers, `d_model=256`, `d_ff=3447`,
`microbatch_size=4`, effective batch size 32, 3,125 nominal optimizer steps,
and a 9,856,480-byte training source.

## Artifact and hash audit

The index `CROSS_GPU_HASHES.json` lists 14 files: four evaluator/source files,
six frozen-input files, and four generated-result/report files. Direct
SHA-256 verification produced:

| Scope | Result |
|---|---:|
| Indexed files present | 14/14 |
| Indexed hashes matching | 14/14 |
| Result JSONs with finite-result flags | 16/16 runs |
| Python source compilation | PASS |

The index intentionally excludes its own self-hash. Its current SHA-256 is
`b6ef60a06272ad1089c49e85d9a07325d1aeb34461b9691659a753ca741032f6`.

The package also contains the unindexed source/data support files required by
the benchmark. Their presence was confirmed. Supplemental hashes, not part of
the frozen index, are:

| File | SHA-256 |
|---|---|
| `context_128/crucible_manifest/stories_train.bin` | `c8d6056d01f91ad391dba99a1b04d48e2b300829c5e6801d910249043bbebf88` |
| `context_128/crucible_manifest/stories_validation.bin` | `0f2ac7dba972bc39b13d7e0dc26d7cf29611c041bdab82956813dad4525eb731` |
| `context_128/source/packed_train.jsonl` | `dca93acdadec204409f7312b6b5b6ea4897bdd846bbd4eb9a62c71b8dce36cce` |
| `context_128/source/packed_validation.jsonl` | `0892f91dc2fee04619ebbd795b7da7f67a7829b9d6423eac1545c50c5512fa23` |
| `context_128/source/provenance_train.jsonl` | `acad7883a82add9f089999b746bb41d1f62404b6e403e1b14bb7d2686ef915e6` |
| `context_128/source/provenance_validation.jsonl` | `b8138bb91388ac22c7477c76e229a5cf5dc3a2b908227614e54d46a4390f4e58` |
| `context_128/source/source_metadata.json` | `b9fdebf9f2e49fd496eea8c4e7b5a9fe94e8098928faf40c6c3c72f94523573c` |

The result package is therefore self-contained for this audit, but the
supplemental data hashes should not be treated as part of the original sealed
index unless a new manifest is deliberately issued.

## Measured performance

The published comparison uses the sum of the eight timed-step `wall_ns`
records. `total_wall_ns` additionally includes warmup, setup and finalization;
it is not the value in the headline timing table.

### Production telemetry enabled

| Seed | Original (s) | Grouped (s) | Reduction | Throughput |
|---:|---:|---:|---:|---:|
| 42 | 25.254401 | 19.159639 | 24.1335% | 1.3181x |
| 1337 | 25.626717 | 19.264643 | 24.8259% | 1.3302x |
| **Mean** | **25.440559** | **19.212141** | **24.480%** | **1.324x** |

### Diagnostic telemetry disabled

| Seed | Original (s) | Grouped (s) | Reduction | Throughput |
|---:|---:|---:|---:|---:|
| 42 | 23.010794 | 16.684452 | 27.4929% | 1.3792x |
| 1337 | 23.369002 | 16.968199 | 27.3901% | 1.3772x |
| **Mean** | **23.189898** | **16.826325** | **27.442%** | **1.379x** |

The 4090 result is portable across the two tested seeds, not necessarily
portable at the same percentage to every GPU. The prior RTX 3060 Ti result was
32.02% with production telemetry, so the 4090 reduction is 7.54 percentage
points smaller while the 4090 itself is faster hardware.

### Timed phase averages, production telemetry enabled

| Phase | Original (s) | Grouped (s) | Reduction |
|---|---:|---:|---:|
| Forward CUDA event scope | 10.776343 | 8.838856 | 17.977% |
| Backward CUDA event scope | 14.419598 | 10.130642 | 29.741% |
| Optimizer and clipping | 0.178506 | 0.178786 | -0.159% |
| CPU batch preparation | 0.017838 | 0.016381 | 8.070% |
| Host-to-device handoff | 0.006790 | 0.006585 | 3.048% |

The saving is concentrated in forward and especially backward CUDA execution,
not in data preparation or optimizer work. These phase scopes are not a
kernel-level attribution and should not be added as independent hardware
costs.

## Numerical-equivalence scope

### What the benchmark does establish

- The same indexed schedule prefix is used by both arms; the first 32 batches
  have identical fingerprints, target counts and raw-byte counts.
- Each arm loads the same seed-specific initial anchor and checks its SHA-256.
- Each arm constructs the same fresh AdamW contract and uses the same learning
  rate, weight decay, betas, epsilon and gradient clipping setting.
- The paired model state hashes match after the eight timed optimizer steps for
  both seeds, both repetitions and both telemetry modes.
- Loss differences are reported as zero within the benchmark's `1e-5`
  comparison tolerance.
- All 16 result runs report finite losses and gradient norms.

### What it does not establish

`portable_benchmark.py:71-79` hashes sorted model `state_dict` entries by name,
dtype, shape and raw tensor bytes. This covers model parameters and buffers. It
does not cover:

- AdamW moment and step tensors;
- scheduler state;
- Python, NumPy, CPU Torch or CUDA RNG state;
- sampler cursor and future schedule state;
- gradient tensors at a checkpoint;
- telemetry accumulators and their serialization;
- process/environment state.

The benchmark creates a fresh optimizer for each arm and resets the Torch seed
before each trial, but it does not serialize those states for comparison. The
current result is therefore **short-run model-state equivalence**, not complete
training-state equivalence.

The grouped implementation in
`grouped_impl/grouped_expert_benchmark.py:226-357` preserves the top-2 route,
weights and expert outputs while executing rank-preserving segments. It still
uses Python expert loops; it is a deterministic grouped execution baseline,
not a GPU-resident or neural scheduler.

The two weighted telemetry reductions at
`shared/model_core.py:506-525` are performed on CPU float64 and copied back to
the existing telemetry buffers. They are intended to be detached observational
statistics. The portable driver uses strict deterministic mode because of this
patch.

## Timing-method audit

The method is credible for a short engineering comparison:

- `cuda_timed` synchronizes before the operation, records CUDA events around
  it, synchronizes the ending event, and records synchronized host time
  (`portable_benchmark.py:138-150`).
- Forward, backward and optimizer timing are separately synchronized.
- Each trial gets a newly loaded model and optimizer.
- Warmup steps are excluded from the headline timed window.
- Telemetry-on and telemetry-off are reported separately.

Limitations that remain:

1. Two warmups and eight timed steps are a small sample for a training claim.
2. Runs are executed in a fixed arm/telemetry order, not randomized or blocked
   by a power/thermal control protocol.
3. No GPU power, clock, temperature or hardware utilization trace is included.
4. CUDA event scopes and host wall scopes are useful paired measurements, but
   they are not a kernel-level causal profile.
5. Synchronous host-to-device copies are used; pinned asynchronous staging is
   not tested.
6. The 4090 benchmark is not a full 3,125-step original-versus-grouped run.

One code-hygiene issue was found: the older standalone
`grouped_impl/gpu_step_benchmark.py` main still searches for arm names
`current_python_stack` and `indexed_precomputed`, while its current variant
set is different. The cross-GPU result was produced by `portable_benchmark.py`,
which uses the correct `original` and `grouped_rank_preserving` names, so this
does not invalidate the sealed 4090 result. The stale entry point should not be
used for future validation until repaired in a new versioned package.

## Memory and scalability

The result JSONs report essentially identical peak memory for the two arms:

| Mode | Allocated peak | Reserved peak |
|---|---:|---:|
| Telemetry off | 3.233098 GiB | 4.000000 GiB on seed 42; 3.978516 GiB on seed 1337 |
| Telemetry on | 3.233363 GiB | 4.000000 GiB on seed 42; 3.978516 GiB on seed 1337 |

These are allocator peaks for this short E16/context-128 workload, not a
general memory requirement. Reserved memory is not the same as live tensor
memory or total physical occupancy.

Relevant scaling constraints are:

- Attention activation/work generally grows with sequence length; the tested
  context is only 128.
- Expert parameter storage grows with expert count if expert width is held
  fixed, while top-2 active expert computation does not grow in the same way.
- Activation memory grows with batch, microbatch and context length.
- Gradient accumulation and checkpointing can change both memory and timing.
- Grouped indexing and temporary buffers may scale differently from the
  original loop, especially under extreme routing skew.

No linear extrapolation to E32/E64, longer context, larger batches, or another
GPU is justified by these measurements.

## Reproducibility and risk assessment

**Strong evidence:** sealed artifact hashes, two seeds, two repetitions, exact
paired model-state hashes, exact input-prefix fingerprints, finite runs, and
the same improvement direction on RTX 3060 Ti and RTX 4090.

**Outstanding risks:** incomplete state hashing, short horizon, fixed benchmark
ordering, hardware-dependent kernel behaviour, possible long-horizon
amplification of tiny numerical changes, and the unverified stale standalone
benchmark entry point.

## Recommendation

Retain rank-preserving grouping as an **opt-in validated execution backend**.
Do not silently replace the original scientific runner and do not claim
long-horizon equivalence yet. The next gate is the staged original-versus-
grouped equivalence test with canonical model, optimizer, scheduler, RNG and
sampler hashes at checkpoints. It should be run on the target GPU only after
the plan is reviewed and explicitly authorized.

No Git commit was made: `/home/timbushnell` is not a Git worktree.
