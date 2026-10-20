# Crucible expert-count parameter/FLOP accounting v1

This is a read-only analytical study. It did not instantiate, modify, train, benchmark, or save a model.

## Frozen reference

- Manifest: `/home/timbushnell/crucible_checkpoint_ab_100k_v2_20261016/portable_manifest/experiment_config.json`
- Manifest SHA-256: `b57bf952599da14f55826e546a4af7cc8e2a9394d1d30d9e0283d5e78456415a`
- Model-core SHA-256: `d5a42f6daa595f1e9e4a53ac2771bae5db5c19820604972592d9659fc1c05b93`
- Reference B/42 checkpoint metadata SHA-256: `49ca621247659a423d4028194b59f1f40e3689dcedd5d43f059785288a380507`
- Reference B/42 checkpoint SHA-256: `98b59bfc2a2b542894dbd3be2436390f1d9de2c97a1c52b0988ab9154f305439`
- Geometry: `4` layers, `d_model=256`, `d_ff=3447`, `E=4`, `top_k=2`, `V=259`, context `128`.

The current implementation uses four local experts, top-2 routing, no capacity limit, no token dropping, and no expert parallelism. Branch B adds its balancing loss during training; this accounting does not change or reimplement that loss.

## Exact parameter formulas

All linear layers have no bias. One SwiGLU expert has three matrices, so:

```text
P_expert_per_layer = E * 3 * d_model * d_ff
P_router_per_layer = d_model * E
P_attention_per_layer =
    d*latent_q + latent_q*d + latent_q*(n_heads*pe_dim)
  + d*latent_kv + 2*latent_kv*d + d*pe_dim + d*d
P_total = 2*V*d + d_final_norm
        + L*(P_attention + 2*d_norm + P_router + P_expert)
P_active = P_nonexpert + L*top_k*3*d_model*d_ff
```

`P_active` uses the usual MoE accounting convention: all dense/non-expert parameters plus only the selected expert parameter sets. It is not a claim that every embedding row is touched by one token; the lookup-row nuance is excluded from the headline comparison.

The reference decomposition is:
- attention weights per layer: `163,840` parameters;
- non-expert base excluding routers: `790,272` parameters;
- reference total: `43,151,104`;
- reference active path: `21,972,736`.

## FLOP formulas and assumptions

The convention is one multiply plus one add = 2 FLOPs. The reported sequence FLOP uses the average causal context for `S=128`; decode FLOPs use a full context-128 KV-cache step.

```text
SwiGLU expert evaluation = 6*d_model*d_ff FLOPs
attention projections = 2*P_attention FLOPs
causal attention score/value work = 4*d_model*(S+1)/2 FLOPs per token on average
router gate = 2*d_model*E FLOPs
router softmax/top-k/renormalization = documented 8*E + 2*top_k proxy
expert weighted combine = 2*top_k*d_model FLOPs
LM head = 2*d_model*V FLOPs
training estimate = 3*forward FLOPs
```

The 3x training multiplier is a planning estimate for forward plus backward, not a measurement. Exact backward cost depends on kernels and autograd saves. RMSNorm and residual arithmetic are included with small fixed proxies. Sin/cos/RoPE, comparisons, indexing, and memory synchronization are not converted into invented FLOPs.

Dispatch is reported separately because it is memory movement rather than stable arithmetic: a lower bound of `4*top_k*d_model` BF16 bytes per token for input/output vector movement, plus `top_k*(8+4)` bytes for int64 expert IDs and FP32 routing weights. The Python mask/gather implementation may exceed this lower bound.

## Memory formulas and assumptions

Runtime code keeps model parameters in FP32 and uses BF16 autocast. The training persistent estimate is `16*P` bytes: FP32 parameters, FP32 gradients, and two FP32 AdamW moments. The activation proxy assumes microbatch 4, context 128, BF16 saved activations, all layers retained for backward, two SwiGLU intermediates per active expert stream, and a conservative attention-score buffer. It is a planning proxy, not an allocator measurement.

Inference memory is FP32 parameter memory plus a one-batch activation proxy. A separate generation KV-cache proxy uses the implementation's compressed KV and positional-key widths. CUDA allocator reserve, kernel workspaces, fragmentation, and telemetry buffers are excluded.

## Regime definitions

### A — fixed total parameters

Expert width is chosen so `E*d_ff` stays equal to the reference. The only total-parameter mismatch is the router output width: fewer experts mean fewer gate weights. This is below 0.01% here, and is reported rather than hidden.

### B — fixed active expert compute

Expert width is chosen so `top_k*d_ff` stays equal to the reference. E=2 with top-2 activates both experts; E=1 necessarily uses top-1. Expert arithmetic is therefore matched, while total capacity and router/dispatch overhead differ.

### C — fixed model geometry

The E=4, d_ff=3447, top-2 Branch B model is unchanged. E=1/2 rows retain d_ff=3447 but are not fair capacity-matched controls: they deliberately have less total expert capacity, and E=1 also has half the active expert compute.

## Implementation feasibility

The constructor accepts a different expert count and width, but the current forward path hard-codes `torch.topk(..., 2)`. E=2 and E=4 are structurally compatible with top-2; E=1 requires a future generalized top-k path and corresponding telemetry/auxiliary-loss handling. This study does not make that code change.

Non-expert dimensions can remain unchanged: embeddings, attention, RMSNorms, output head, context, and optimizer schedule. The dimensions that must change are expert count, router output width, expert hidden width, and top-k for E=1. Branch-B balancing semantics must be explicitly generalized before training variants.

## Table

See `comparison_table.csv` and `comparison_table.json` for machine-readable values. The table intentionally exposes parameter, active-compute, router, dispatch, and memory mismatches separately.
