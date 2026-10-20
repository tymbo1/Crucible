# Sustained grouped production validation

Completed 2026-10-19 (Australia/Brisbane).

The corrected grouped-rank-preserving path completed the prescribed 3,125-update schedule for both seeds with the production telemetry correction enabled.

## Integrity

- expected records: 18
- observed records: 18
- seeds: 42 and 1337
- checkpoint steps per seed: 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125
- duplicate or missing cells: none
- non-finite candidate records: none
- interval NaN/Inf count: zero throughout
- layer telemetry: all four MoE layers present in every record
- validation metric: EOS-excluded raw-byte NLL/BPB
- manifest SHA-256: `3d7b4e4e3ce976004df88e78b8588456c974b40a38e57f8cba79583f1f3695d0`

## Final records

| Seed | Validation NLL | Raw-byte NLL | BPB | Routing CV | Load ratio | Peak CUDA memory |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.755284 | 0.758363 | 1.094086 | 0.033062 | 1.125416 | 3,422,476,800 bytes |
| 1337 | 0.750260 | 0.753947 | 1.087715 | 0.038555 | 1.137422 | 4,777,753,600 bytes |

Recorded training interval totals were 2,923.837558 seconds for seed 42 and 2,905.254351 seconds for seed 1337. The corresponding validation totals were 437.820627 and 433.968647 seconds. Each seed processed 9,856,480 source bytes and the same number of raw-byte targets.

This is a grouped-only sustained validation, not a paired 3,125-step original-versus-grouped equivalence run. It therefore confirms clean completion and telemetry health for the corrected grouped path, but it does not claim a sustained relative speedup or long-horizon equivalence against the original implementation.

The raw JSONL and metadata are the authoritative records. Their hashes are recorded in the package-level seal index.
