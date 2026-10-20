# Crucible pretrained-foundation comparative plan

Date: 2026-10-21
Status: design-only; no checkpoints downloaded, no model modified, no training or cloud rental authorized.

## Repository audit

The Crucible artifact repository is available remotely:

- URL: `https://github.com/tymbo1/Crucible-artifacts`
- Verified SSH ref: `main` / `4135fe430f88a76a6c679e6fc3c1da11aeb92901`
- Role: archived experiment outputs and reproducibility artifacts; not treated as the Crucible source worktree.

The nearest tracked source repository is:

- Path: `/home/timbushnell/braidformer_lab`
- Remote: `git@github.com:tymbo1/Braidformer.git`
- Branch: `main`, locally ahead of `origin/main` by 120 commits at audit time
- Crucible source files: not present in the tracked file set

The active Crucible implementation is stored in standalone, versioned experiment directories, including:

- `/home/timbushnell/crucible_v2_production_telemetry_v1_20261019/`
- `/home/timbushnell/crucible_context_gate_v1_20261018/`
- `/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018/`
- `/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018/`

Those directories are not Git worktrees. No separate Crucible source repository was found locally; the remote `Crucible-artifacts` repository is an artifact archive, not a source checkout. No repository was initialized, and no files were moved.

## Reference geometry

The E16 Branch B reference has 170,233,600 trainable parameters, a 259-ID lossless byte vocabulary, width 256, four layers, four attention heads, custom latent attention, 16 top-2 SwiGLU experts per MoE layer, and expert hidden width 3,447.

The current model is not a standard Llama/Qwen/OLMo Transformer. Its attention uses separate latent query/KV compression and expansion projections plus a 32-dimensional rotary-style branch. The tokenizer, output head, router, expert blocks and attention geometry therefore require explicit compatibility treatment.

## Candidate comparison

The official candidate configurations are materially different from Crucible:

| Candidate | Published geometry relevant here | Transfer implication | Pilot cost |
|---|---|---|---|
| SmolLM2-135M | Llama architecture; 30 layers; width 576; intermediate 1,536; 9 attention heads; 3 KV heads; vocabulary 49,152 | Closest parameter scale and uses a SwiGLU-style Llama FFN, but no direct width, layer, attention or vocabulary match | Lowest |
| Qwen3-0.6B-Base | 28 layers; width 1,024; intermediate 3,072; 16 attention heads; 8 KV heads; vocabulary 151,936 | Much larger and architecturally incompatible with the 256-wide latent-attention Crucible core | Medium/high |
| OLMo 2 1B | 16 layers; width 2,048; intermediate 8,192; 16 attention heads; vocabulary 100,352 | Largest mismatch and highest memory/teacher-inference cost, but attractive later for open research provenance | Highest |

Sources: [SmolLM2 configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json), [Qwen3-0.6B-Base configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json), and [OLMo 2 configuration](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json).

All three are candidates for research planning, not certified drop-in initializations. Their tokenizer vocabularies differ from Crucible's 259 byte IDs, so native embedding and LM-head transplantation is not valid.

## Route A — partial pretrained FFN transplantation

### Compatibility

This route can only transfer a subset of source tensors. A source FFN would need compatible activation semantics and a documented mapping into Crucible's `256 -> 3447 -> 256` SwiGLU geometry. None of the three candidates has an exact shape match.

The possible mappings are:

1. direct copy only where shapes and conventions match;
2. fixed projection/padding/truncation into Crucible dimensions;
3. learned adapter projections trained before or alongside the experts.

Options 2 and 3 are not pure weight transplantation and must be labelled as adapted initialization.

### Experimental confounds

- The attention, embeddings, LM head and routers remain newly initialized.
- Cloning one dense FFN into 16 experts creates expert symmetry.
- Random expert perturbations break exact function preservation.
- A projection adapter adds new capacity and can receive credit for the result.
- Any vocabulary conversion makes the initial model's language function non-equivalent.

### Cost and memory

After the source tensors are available, training cost is approximately the ordinary Crucible cost plus one-time mapping. The main additional cost is engineering and source-weight storage, not teacher inference. The current E16 model fits the existing 8-GB class under the tested settings; mapped source weights should be discarded after conversion unless retained for audit.

### Pilot success criteria

- Shape and dtype mapping report has zero silent truncations.
- All unmapped tensors are explicitly listed.
- No NaN/Inf events and exact data-order fingerprints.
- At 100 and 500 updates, the transplanted arm is compared with the same-seed from-scratch control.
- Promotion requires a mean BPB improvement of at least `0.005` at 500 updates and no seed worse than the scratch control by more than `0.002`, or reaching the scratch control's BPB with at least 20% fewer raw training bytes.

## Route B — teacher-student distillation

### Compatibility

Distillation leaves Crucible's architecture and byte tokenizer unchanged. It avoids tensor-shape transplantation, but tokenizer mismatch prevents a naive token-by-token KL loss.

The first implementation should use sequence-level or text-continuation distillation: the teacher receives detokenized text and supplies a continuation or ranked targets; Crucible learns the corresponding UTF-8 byte sequence. A future hidden-state or boundary-aligned logit loss can be considered only after its alignment is specified.

### Experimental confounds

- Teacher-generated text changes the data distribution unless the original corpus remains the primary source.
- Sequence-level distillation can measure imitation of teacher outputs rather than transferred competence.
- Teacher inference cost must be reported separately from student training cost.
- Native teacher-token loss is not directly comparable with Crucible byte-level BPB.

### Cost and memory

Approximate BF16 parameter storage before framework overhead is 0.27 GB for a 135M teacher, 1.2 GB for a 0.6B teacher and 2.0 GB for a 1B teacher. A frozen teacher needs no optimizer state, but its forward activations and student training state add memory. SmolLM2 is the only candidate that should be considered first on the 8-GB local GPU; Qwen3 may require careful microbatching; OLMo 2 should be deferred to larger hardware or offline teacher generation.

### Pilot success criteria

- Same Crucible architecture, tokenizer, data split, raw-byte budget, optimizer and seeds as the scratch control.
- Teacher generation/inference time reported separately.
- Student has no non-finite events and retains exact schedule/provenance integrity.
- Promotion requires the same `0.005` mean BPB improvement / `0.002` worst-seed regression thresholds as Route A, or the 20%-less-byte criterion.
- A second evaluation on untouched held-out stories must confirm the gain; teacher imitation alone is insufficient.

## Route C — from-scratch baseline

The existing E16 Branch B from-scratch training remains the control. It must use the frozen tokenizer, manifests, seeds, optimizer contract and grouped backend only after the sustained-metric validation qualification already documented.

The baseline is rerun only as part of an explicitly authorized matched pilot; prior frozen results remain unchanged.

## Recommended first pilot

Choose **SmolLM2-135M as a frozen teacher for Route B**, not as a direct tensor transplant.

Reasoning:

- It is closest to Crucible's total scale.
- Its official configuration is small enough for a low-cost teacher-inference pilot.
- Distillation avoids pretending that incompatible attention, vocabulary and layer tensors are reusable.
- It gives a cleaner first test of whether pretrained language competence can improve the unchanged Crucible architecture.
- The FFN-transplant route should follow only if a configuration-only mapping and synthetic state-dictionary test show a defensible transfer boundary.

OLMo 2 1B is the best later provenance-rich teacher candidate, not the first pilot: its larger geometry and memory cost are unnecessary until the method works at small scale. Qwen3-0.6B-Base is a useful intermediate teacher after the SmolLM2 pilot, but its 151,936-token vocabulary and 1,024-wide core make it a poor first transplantation target.

## Proposed controlled pilot

### Cells

| Arm | Description |
|---|---|
| A | E16 Crucible from scratch |
| B | E16 Crucible with documented FFN transplantation, only if mapping audit passes |
| C | E16 Crucible distilled from frozen SmolLM2-135M teacher |

Use two seeds, 100-update mechanical preflight, then 500-update pilot. Match raw-byte exposure and optimizer updates. Keep context 128, E16/top-2 routing, Branch B loss and validation construction fixed.

### Acceptance gates

1. **Integrity:** exact manifest/config hashes, sample-order fingerprints, finite loss/gradients, and checkpoint readability.
2. **Transfer signal:** candidate beats the scratch control by at least `0.005` mean BPB at 500 updates with no seed regression greater than `0.002`, or reaches the control BPB using at least 20% fewer raw bytes.
3. **Generalization:** the result holds on the untouched validation split and is not confined to teacher-generated material.
4. **Cost:** teacher inference and any mapping overhead are reported; a small BPB gain that costs more than its saved training compute does not pass the engineering gate.

### Budget envelope

- This design and compatibility audit: zero GPU cost.
- Configuration-only and synthetic mapping tests: zero GPU cost.
- Local 100/500-update pilot: no cloud cost; runtime to be measured before any longer run.
- If cloud execution is later approved: reserve a maximum of **US$5** for the six-cell pilot on a secure RTX 4090, with a hard stop before exceeding the cap. Teacher inference may require a separate cap if the first preflight shows the 8-GB local GPU is unsuitable.

This is a planning envelope, not authorization to spend.

## Unresolved questions before execution

- Should distillation use teacher-generated continuations, teacher ranking, or a boundary-aligned objective?
- What exact SmolLM2 tokenizer-to-UTF-8 procedure will define teacher inputs and outputs?
- Should the FFN route use fixed projections or learned adapters? This changes the causal interpretation.
- Should all Crucible parameters be trainable from update 1, or should transferred components be briefly frozen?
- What teacher data is allowed without contaminating the held-out evaluation split?
- Should the grouped backend be used in the pilot, or should the original backend be retained for the first comparison to minimize engineering confounds?

No execution should begin until these questions are preregistered in the pilot manifest.
