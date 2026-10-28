# Crucible BraidFormer architecture audit

Date: 2026-10-29

## Decision

Do not replace the current Crucible dense latent-attention backbone with
BraidFormer A1 for the foundation run. Keep BraidFormer as a separately
versioned comparison and possible later ablation.

The current foundation question is whether Crucible can acquire language and
reasoning competence on the selected data mixture. Changing the backbone at
the same time would confound that question with a second architecture
hypothesis.

## What was inspected

The local BraidFormer repository is at:

`/home/timbushnell/braidformer_lab`

Audited repository commit: `53545a41360a0588a6bc66afc3e370e9aaa5924b`.

Relevant implementation and evidence:

- `src/braidformer/models/transformer.py`: A0 conventional decoder control.
- `src/braidformer/models/recurrent.py`: A1 gated diagonal state-space-majority
  decoder with associative scan support.
- `docs/A0_A1_EIGHT_EQUAL_CUDA_RESULT.md`: three-seed matched-CUDA comparison.
- `docs/A1_TRAINING_RESULT.md`: seed-17 training and equal-token comparison.
- `docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md`: mechanism-level failure audit.
- `docs/TRAJECTORY_AWARE_TRAINING_PROPOSAL.md`: deferred BraidFormer
  trajectory-aware training-policy proposal.

## Evidence summary

The BraidFormer A0/A1 family is experimentally more mature than a purely
speculative design. A1 has CPU construction tests, CUDA scan checks,
checkpointing and matched-CUDA experiments. In the three-seed 8x comparison,
A1's validation loss was lower than A0's in every seed, with mean final delta
`A1 - A0 = -0.245285` under the declared TinyStories matched-CUDA protocol.

That result is narrow. A1 processed fewer tokens and had approximately half
the measured token throughput of A0. It is evidence about a small byte-level
TinyStories task, not broad reasoning, cumulative learning or general
architectural superiority.

The mechanism adjudication is equally important: both A0 and A1 fit training
associations while held-out compositional binding degraded with distance and
association count. A1 added a stronger temporal/state penalty. Long-decay
channels and simple state replay did not repair the failure. The best current
description is distributed recurrent transformation/interference combined
with temporal retention and binding demands, not a solved reasoning mechanism.

The trajectory-aware proposal is different again. It predicts batch utility
from detached training-history summaries and may gate or downweight updates.
It is not a reasoning backbone, memory system or recursive answer mechanism,
and its own document marks implementation and training as deferred.

## Compatibility with current Crucible

| Question | Finding |
|---|---|
| Does A1 provide a plausible alternative backbone? | Yes, as a controlled research arm. |
| Has A1 established better general reasoning? | No. Its evidence is TinyStories-focused and its relational audit remains negative/inconclusive. |
| Does A1 implement the recovered OP controller? | No. OP is a separate recursive controller specification. |
| Does the trajectory policy implement cumulative learning? | No. It is a deferred adaptive-compute proposal. |
| Should it replace the current dense backbone during foundation training? | No; that would confound foundation competence with backbone selection. |
| Should it be retained for a later comparison? | Yes, after the dense baseline reaches measurable language competence. |

## Comparison for GPT-style training

Both systems are autoregressive decoder families, but they answer different
questions:

| Aspect | BraidFormer A0/A1 | Current Crucible dense line |
|---|---|---|
| A0/A1 role | A0 is a conventional Transformer control; A1 replaces most sequence processing with a gated recurrent/state-space path. | Custom dense causal decoder using latent attention and SwiGLU blocks. |
| Input/output | BraidFormer's TinyStories pilot used a 32k-token byte-level BPE interface. | Current pilot uses a 259-ID lossless byte interface. |
| Main architectural hypothesis | Recurrent/state-space retention plus later selective retrieval and conditional compute. | Causal latent attention plus dense feed-forward transformation, with OP currently external. |
| Evidence scale | Approximately 42M parameters, three-seed matched-CUDA TinyStories comparisons. | Current foundation pilot is 3.936M parameters on FineWeb-Edu plus reasoning samples; it is a baseline, not a scale comparison. |
| GPT suitability | A useful experimental GPT alternative, but A1's recurrent binding boundary must be carried forward. | The cleanest current Crucible baseline for measuring language acquisition and OP compatibility. |
| Training-policy controller | BraidFormer's trajectory-aware controller is a separate deferred meta-training proposal. | No trajectory controller is active in the foundation run. |

For a GPT-like research trainer, the sensible order is therefore: establish a
competent Crucible dense baseline, compare it with BraidFormer A0 under matched
data and compute, then test BraidFormer A1 as a distinct recurrent hypothesis.
Do not merge A1, selective attention, MoE, adaptive depth and the trajectory
controller into one model before their individual contributions are measured.

## Recommended sequence

1. Train the current Crucible dense latent-attention + SwiGLU model on a
   properly recorded language foundation mixture and the five reasoning
   sources.
2. Establish held-out language loss, generation quality and domain metrics.
3. Compare one-pass reasoning with the exact bounded OP external loop.
4. Only after that baseline is competent, run a matched A0/A1 BraidFormer
   comparison on the same packed data and compute budget.
5. Treat the trajectory-aware policy as a later training-efficiency arm, with
   its own controls for loss gating, audit floors, false skips and controller
   overhead.

This preserves the value of the BraidFormer work without treating a narrow
matched-CUDA result as permission to replace Crucible's backbone before the
current foundation has learned anything substantive.
