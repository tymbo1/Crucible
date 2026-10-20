# Crucible V2 production-telemetry validation

Status: completed 2026-10-19 (Australia/Brisbane)

This package is an isolated validation of the grouped-rank-preserving execution path with the surgical telemetry correction. Frozen Crucible packages, manifests, anchors, schedules, datasets and prior results were not modified.

## Configuration

- Branch: B
- Context: 128 bytes
- Experts: 16
- Routing: top-2
- Seeds: 42 and 1337
- Train updates in sustained validation: 3,125 per seed
- Trainable parameters: 170,233,600
- Manifest binding SHA-256: `3d7b4e4e3ce976004df88e78b8588456c974b40a38e57f8cba79583f1f3695d0`
- Execution variant: `grouped_rank_preserving`

The new `model_core.py` changes only the two selected-probability weighted CUDA reductions used for telemetry. They are detached, reduced on CPU in float64, and copied back to the telemetry accumulator. Model/autograd tensors are not changed by this correction.

## 100-step production equivalence gate

The corrected gate passed for both seeds, comparing original and grouped execution through 100 updates:

- model-state hashes: exact
- gradient hashes: exact
- optimizer-state hashes: exact
- four-layer telemetry coverage: present
- maximum affinity difference in the corrected comparison: zero
- finite training and validation metrics: pass

The raw gate report is preserved. Its initial acceptance flag was invalidated by comparator defects for vector-valued fields and equal non-finite values; the corrected report fixes the comparator and is the authoritative gate result. The model execution itself was not rerun or changed for that correction.

## Paired execution benchmark

Means across two seeds and two repetitions; each cell used two warm-ups and eight timed optimizer steps. Times are the primary timed window, not total process wall time.

| Execution | Telemetry | Mean timed seconds |
|---|---:|---:|
| Original | off | 12.238298 |
| Original | patched production telemetry | 13.683741 |
| Grouped rank-preserving | off | 7.711725 |
| Grouped rank-preserving | patched production telemetry | 9.301640 |

Derived results:

- patched telemetry overhead on original: `+1.445442 s`, `+11.8108%`
- patched telemetry overhead on grouped: `+1.589915 s`, `+20.6169%`
- grouped speedup without telemetry: `36.9870%` (`1.58697x`)
- grouped speedup with required patched telemetry: `32.0241%` (`1.47111x`)
- production-telemetry speedup by seed: 31.8622% (42) and 32.1857% (1337)

The first seed-42 benchmark attempt exhausted VRAM in the harness during its second repetition because wrapper objects were retained. It is preserved and excluded. Explicit garbage collection was added; the retry and the seed-1337 run completed with finite metrics and exact final-state hashes.

## Sustained grouped validation

The sustained run completed the exact expected Cartesian checkpoint set: 2 seeds × 9 checkpoints = 18 records.

Checkpoints: `100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125`.

Integrity checks:

- records: 18/18
- duplicate cells: none
- unexpected or missing cells: none
- non-finite candidate records: none
- `interval_nan_inf_count`: 0 throughout
- context, expert count, top-k and variant: constant across records
- all four MoE layer telemetry containers: present
- EOS-excluded raw-byte validation: enabled

| Seed | Training interval seconds | Validation seconds | Source bytes | Mean train bytes/s | Final validation NLL | Final raw-byte NLL | Final BPB | Final routing CV | Final load ratio | Peak CUDA bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2923.837558 | 437.820627 | 9,856,480 | 3371.08 | 0.755284 | 0.758363 | 1.094086 | 0.033062 | 1.125416 | 3,422,476,800 |
| 1337 | 2905.254351 | 433.968647 | 9,856,480 | 3392.64 | 0.750260 | 0.753947 | 1.087715 | 0.038555 | 1.137422 | 4,777,753,600 |

This was a grouped-only 3,125-step production validation. It establishes that the corrected grouped path completed the full schedule cleanly; it is not a 3,125-step original-versus-grouped equivalence comparison and does not by itself establish a sustained 32% speedup. The 32.0241% figure is the paired eight-step benchmark result above.

## Boundaries and next decision

The production correction is now suitable for the next explicitly authorized sustained original-versus-grouped comparison. No such paired 3,125-step comparison was launched in this package. The two weighted reductions were isolated as a causal pair, but this package does not distinguish the individual contribution of each reduction or fully separate CPU-reduction effects from synchronization effects.

No RunPod or other cloud compute was used for this validation. GPU health was normal after the earlier transient `nvidia-smi` communication failure; no Xid/reset/OOM event was observed during the completed production run.

## Key artifacts

- [Measurement protocol](MEASUREMENT_PROTOCOL.md)
- [Sustained validation report](sustained_3125/SUSTAINED_VALIDATION_REPORT.md)
- [Corrected 100-step equivalence report](equivalence_100/equivalence_report_corrected.json)
- [Corrected 100-step equivalence hashes](equivalence_100/equivalence_hashes_corrected.json)
- [Seed-42 paired benchmark](overhead_seed42_retry1/gpu_step_benchmark_summary.json)
- [Seed-1337 paired benchmark](overhead_seed1337/gpu_step_benchmark_summary.json)
- [Sustained metrics](sustained_3125/metrics.jsonl)
- [Run metadata](sustained_3125/run_metadata.json)
- [Seal hash index](PRODUCTION_TELEMETRY_SEAL_HASHES.json)
