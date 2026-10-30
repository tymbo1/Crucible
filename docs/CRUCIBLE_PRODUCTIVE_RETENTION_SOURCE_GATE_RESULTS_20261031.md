# Productive-retention source-competence gate

Date: 2026-10-31  
Classification: **INVALID/INCONCLUSIVE prerequisite failure**

The authorized source phase completed for both registered seeds. It did not
meet the source-competence requirement, so no C0--C3 target training or sealed
target evaluation was started.

| Seed | Update-250 | Update-500 | Update-750 | Update-1000 | Gate |
|---:|---:|---:|---:|---:|:---|
| 42 | 10/16 | 7/16 | 8/16 | **6/16** | Fail |
| 1337 | 9/16 | 8/16 | 11/16 | **11/16** | Fail |

The registered threshold was at least 13/16 in each seed. The two-seed result
therefore does not establish source competence and does not falsify productive
retention. It blocks interpretation of the transfer experiment exactly as
specified in the protocol.

## Controlled execution

- Starting checkpoint: step 15,000; base SHA-256 recorded in the source-gate reports.
- Source updates: 1,000 per seed.
- Source examples: 8,000 per seed.
- Model: 120M Crucible dense latent-attention + SwiGLU.
- Precision: BF16; microbatch 8; learning rate `3e-4`.
- Data: public source-train and source-validation files only.
- Target data loaded: no.
- Sealed target answers loaded: no.
- C0--C3 target training: not started.

The source prompt encoding and fine-tuning settings are recorded in the
machine-readable reports. They are part of this bounded result and must not be
silently changed while interpreting it.

## Checkpoint integrity

Both update-1,000 source checkpoints were reloaded independently. All 185
floating-point model tensors and 555 floating-point optimizer tensors were
finite in each checkpoint.

| Seed | Final source checkpoint SHA-256 |
|---:|:---|
| 42 | `7591947e5b316e46fed45f2a9f806370750c0a0d051245ad62dd89a4dbdb3612` |
| 1337 | `261265938b6b356cabf4382ce862382c3a5e2aaf94593f3e6ce9a32d8771724d` |

## Interpretation

This is a prerequisite failure, not evidence against cumulative learning.
Possible causes include insufficient source adaptation, the Boolean prompt/task
interface, or optimization behavior, but this run does not distinguish them.
The correct next decision is to freeze this negative gate result and perform a
separate novelty audit before proposing any source-learning change. The
productive-retention target experiment remains blocked.
