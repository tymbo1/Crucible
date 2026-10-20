# Crucible pretrained-initialization feasibility assessment

Date: 2026-10-21
Status: CPU-only design study; no checkpoints downloaded and no training started.

## Reference architecture inspected

Sources inspected:

- `crucible_v2_production_telemetry_v1_20261019/shared/model_core.py`
- `crucible_context_gate_v1_20261018/context_128/crucible_manifest/experiment_config.json`

The current E16 Branch B geometry is:

| Component | Value |
|---|---:|
| Vocabulary | 259 lossless byte-level IDs |
| Model width | 256 |
| Layers | 4 |
| Attention heads | 4 |
| KV/query latent widths | 64 / 64 |
| Positional projection width | 32 |
| Experts per layer | 16 |
| Routing | top-2 |
| Expert hidden width | 3,447 |
| Context window | 128 bytes |
| Trainable parameters | 170,233,600 |

Each block is pre-norm and contains custom `MultiHeadLatentAttention` followed by a residual `ResearchMoELayer`. Each expert is a bias-free SwiGLU with matrices `256x3447`, `256x3447` and `3447x256`. The router is a bias-free `256x16` projection.

The attention implementation is not ordinary packed QKV attention. It uses separate query/KV compression and expansion projections plus a 32-dimensional rotary-style positional projection. There are no learned positional-embedding tensors in the inspected implementation.

## Parameter accounting

| Component | Parameters |
|---|---:|
| Token embedding `259x256` | 66,304 |
| One expert SwiGLU | 2,647,296 |
| 16 experts in one layer | 42,356,736 |
| Router in one layer | 4,096 |
| Latent attention in one layer | 163,840 |
| Two RMSNorm weights in one layer | 512 |
| One complete layer | 42,525,184 |
| Four layers | 170,100,736 |
| LM head `256x259` | 66,304 |
| Final RMSNorm | 256 |
| **Total** | **170,233,600** |

The accounting matches the manifest's expected parameter geometry.

## Weight-transfer compatibility

| Crucible component | Direct reuse from an arbitrary pretrained Transformer? | Reason |
|---|---|---|
| Token embedding | No, unless the source has the same 259-ID byte vocabulary and ID semantics | BPE/SentencePiece token rows cannot be copied into byte IDs without a mapping or new initialization |
| LM head | No, for the same vocabulary reason | Output rows have different semantic units |
| Latent attention | Usually no | Standard QKV projections do not map directly to Crucible's separate latent projections and positional branch |
| RMSNorm | Conditional | Reusable only when width, normalization type, placement and parameter convention match |
| Dense FFN to one expert | Conditional | Requires compatible SwiGLU orientation and compatible hidden width; otherwise needs projection, padding, truncation or distillation |
| Dense FFN to all 16 experts | Not function-preserving by itself | Cloning creates expert symmetry; perturbations or differentiated training are required |
| MoE router | No | Crucible's router has no pretrained dense-model counterpart |
| Expert bias-tracking buffers | No | These are Crucible runtime state, not pretrained language weights |

The most important blocker is not parameter count. It is representation compatibility: the byte-level input/output interface and the custom latent attention geometry are both different from the usual open-weight language-model interfaces.

## Feasible initialization paths

### 1. Strict tensor transplant

Low feasibility for the named candidates without an exact architecture match. This path is appropriate only after a candidate's configuration and tensor names have been audited. It should not be attempted by shape-only copying.

### 2. Partial transplant into matched submodules

Potentially feasible for normalization and selected FFN tensors if dimensions and mathematical conventions match. It would not initialize the tokenizer interface, router or custom attention completely, so the result must be called partial initialization.

### 3. Dense-to-MoE initialization

Potentially useful if a source SwiGLU FFN can be mapped. A dense FFN can seed each expert, but identical copies create symmetry. Expert-specific perturbations change the function immediately, so this is not a function-preserving conversion. The router must be newly initialized and trained.

### 4. Teacher distillation

The most generally feasible route when tokenizer and attention geometry do not match. A pretrained teacher supplies targets or hidden-state supervision while Crucible remains byte-level. This reuses learned competence without claiming that source tensors were transplanted.

## Candidate status

SmolLM2-135M, Qwen3-0.6B-Base and OLMo 2 1B remain planning candidates only. Their checkpoints, configurations, tokenizer files and licenses were not downloaded or audited in this CPU-only study. None is currently certified as a drop-in Crucible initialization.

## Recommended future experiment

Use four clearly separated arms:

1. Current E16 Crucible from scratch.
2. Unmodified pretrained source baseline, evaluated under its native tokenizer and metrics.
3. Crucible with an explicitly documented partial transplant or distillation initialization.
4. A matched random-initialized Crucible control with identical data, seeds, optimizer updates and compute budget.

The primary comparison for Crucible should use its byte-level BPB/BPC evaluation, fixed data exposure and identical training schedule. Native-tokenizer source-model loss should be reported separately; it is not directly comparable to byte-level loss.

Required measurements include initial loss, loss after fixed byte/token budgets, validation BPB, convergence speed, trainable/frozen parameter counts, adapter/transplant coverage, peak memory, throughput and seed-to-seed variance.

## CPU-only next steps

1. Build a configuration-only compatibility checker for candidate model metadata.
2. Require exact tokenizer, normalization, attention and FFN mapping reports before any weight download.
3. Use synthetic state dictionaries to test mapping and shape checks.
4. Decide between partial transplant and distillation only after the audit.
5. Request checkpoint download and GPU work as a separate authorization.

No checkpoint download, architecture modification, fine-tuning or GPU benchmark is authorized by this assessment.

