# Crucible context-gate collection audit

Date: 2026-10-21  
Scope: read-only audit of the collected RunPod metric files and the frozen context manifests.

## Verdict

The context collection is structurally valid but incomplete.

| Context | Expected seeds | Observed rows | Observed seeds | Status |
|---:|---|---:|---|---|
| 128 | 42, 1337 | 18 | 42, 1337 | Complete |
| 256 | 42, 1337 | 9 | 42 | Partial; all nine seed-1337 checkpoints missing |
| 512 | 42, 1337 | 9 | 42 | Partial; all nine seed-1337 checkpoints missing |

Every context manifest expects checkpoints at:

`100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125`

The 128 file has all nine checkpoints for both seeds. The 256 and 512 files
have all nine checkpoints for seed 42 and none for seed 1337. No malformed or
duplicate metric rows were found in the collected files.

## Frozen configuration

All three variants use the E16 Branch B configuration:

- 16 experts, top-2 routing;
- `d_model=256`, `d_ff=3447`, four layers;
- byte-level vocabulary of 259;
- 3,125 optimizer updates;
- dynamic source-byte-balanced step groups;
- EOS-excluded raw-byte NLL/BPB as the primary endpoint;
- expected seeds 42 and 1337;
- target source bytes per update approximately 3,154 bytes.

The context-specific target source bytes per update are 3,154.0736 (128),
3,154.50272 (256) and 3,154.95552 (512). The manifests therefore preserve
the intended byte-exposure control, subject to their documented small packing
residuals.

## Collected metric hashes

These hashes match the collection-status record:

| Context | File | Rows | SHA-256 |
|---:|---|---:|---|
| 128 | `context_128/metrics.jsonl` | 18 | `5dd005c1cf06f3633954733941b2a32d2d883a6dd5450d04dd024ed3beb6777e` |
| 256 | `context_256/metrics_seed42.jsonl` | 9 | `93996de7b5a862ded2be19724e70c159b9237c4d831d61ddfe76bd9ae9463f07` |
| 512 | `context_512/metrics_seed42.jsonl` | 9 | `f2499fba0aa27a13a2aec8e228ce090afb35440bc668dba8c51c9d6700447e82` |

The corresponding frozen experiment-config hashes are:

| Context | Config SHA-256 |
|---:|---|
| 128 | `15ed67b3e38db7a82ea6dcf7825f0e51c6ff2c66bd31da9aeead67aa3a309a72` |
| 256 | `43e23eb51466f3b00e53efe74756a512fefb0d63a7d5f995496504b03b3cc1bc` |
| 512 | `3c12c9992df2af799369b7ce86cd9f61068d681b779cf8041ef6631996092980` |

## Observed final seed-42 metrics

These are descriptive observations only. They do not support a context-length
conclusion until seed 1337 is available for 256 and 512.

| Context | Final step | Validation raw-byte NLL | BPB |
|---:|---:|---:|---:|
| 128 | 3125 | 0.7566858501 | 1.0916669234 |
| 256 | 3125 | 0.7674872475 | 1.1072500459 |
| 512 | 3125 | 0.8105088218 | 1.1693170578 |

The complete 128 seed-1337 final row is raw-byte NLL `0.7551713873` and BPB
`1.0894820155`.

## Reproducible audit

The audit implementation is:

`tools/audit_context_gate.py`

It checks JSON validity, required fields, context and variant identity,
configuration hashes, duplicate `(seed, checkpoint)` keys, expected checkpoint
coverage and the collection-status SHA-256 values. Its generated report is:

`experiments/crucible_context_gate_v1_20261018/context_audit_20261021.json`

The audit returns success for structurally valid files even when the collection
is incomplete; incompleteness is the result being reported, not a parser
failure.

## Decision

Do not pool 256/512 with the complete 128 matrix or claim a context scaling
result. Preserve the captured files unchanged. A future paid run should first
complete the two missing seed-1337 arms, unless a later decision explicitly
replaces this context question with a more valuable state-saving campaign.
