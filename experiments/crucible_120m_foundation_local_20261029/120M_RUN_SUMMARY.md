# Crucible 120M local foundation run

Classification: `substantive_local_foundation_training`

## Configuration

The existing dense Crucible latent-attention + SwiGLU architecture was scaled
to exactly `120,842,496` parameters:

| Setting | Value |
|---|---:|
| Layers | 14 |
| Model width | 768 |
| Attention heads | 12 |
| Latent KV/Q width | 192 / 192 |
| Positional branch | 64 |
| SwiGLU width | 3,072 |
| Context | 512 |
| Micro-batch / accumulation | 2 / 4 |
| Effective batch | 8 |
| Activation checkpointing | Enabled |

## Measured result

The run completed 5,000 optimizer updates from scratch on the RTX 3060 Ti,
with a resumable continuation from step 1,000.

- Training tokens: `8,425,867`
- End-to-end local throughput, including both segments: approximately
  `3,039 tokens/s`
- Continuation-segment throughput: `4,078 tokens/s`
- Peak allocated VRAM: `2.93 GB`
- Peak reserved VRAM: `3.33 GB`
- Peak resident CPU memory: `2.05 GB`
- Non-finite losses/gradients: none
- Checkpoints: steps 250, 500, 750, 1,000, 2,000, 3,000, 4,000 and 5,000

FineWeb validation loss decreased from `4.8902` at step 1 to `1.5398` at
step 5,000. Reasoning validation loss fluctuated around `2.5–2.9`; this does
not yet establish broad reasoning competence. Greedy generations show partial
language structure but remain noisy and repetitive.

## Local time estimates

Using the measured end-to-end rate, excluding future scaling effects:

| Token budget | Estimated local time |
|---:|---:|
| 100M | 9.1 hours |
| 500M | 45.7 hours |
| 1B | 91.4 hours |

These are engineering estimates from one local run, not promises. They include
the current checkpointing and evaluation pattern and can change with batch,
sequence length, data mix, or thermal state.

## Interpretation

The local machine can train this 120M configuration without CPU optimizer
offload. Offloading is therefore not justified for this configuration unless
a later larger-context or larger-batch test shows a memory constraint.

The current result establishes a viable 120M training path, not a finished
language model or evidence that OP improves reasoning. OP remains available
for a later checkpoint comparison; repeated OP evaluations were deliberately
not inserted into this foundation run.

Raw FineWeb data, packed arrays and model checkpoints remain outside Git under
the local data/checkpoint roots. Sanitized metrics and hashes are committed.
