# GPU Neural SSRE v0 — attribution gate

Date: 2026-10-18  
Device: NVIDIA GeForce RTX 3060 Ti  
Input: frozen Crucible E16 Branch-B, context 128, seed 42  
Profile: one warmed schedule step, nine microbatches, forward/backward only

## Integrity

- CUDA profile completed successfully.
- All profiled losses were finite.
- 3,245 valid targets and 3,212 raw-byte targets were consumed.
- Optimizer steps performed: 0.
- Initial and final model-state SHA-256: `e99d193d671ef4080042285077bdff1dea4e752853bc06d69e296215345133ad`.
- Model state unchanged: yes.
- Peak allocated VRAM: approximately 1.714 GiB.
- Profiled forward/backward CUDA event: approximately 2.634 s. This is a
  profiler measurement, not a throughput benchmark.

## Expert workload

Each layer routed 6,490 top-2 expert assignments, exactly twice the 3,245
valid targets. All 16 experts received traffic in all four layers, but the
single-step distribution was not uniform:

| Layer | Minimum selected tokens | Maximum selected tokens | Nonzero experts |
|---:|---:|---:|---:|
| 0 | 24 | 1,372 | 16 |
| 1 | 102 | 820 | 16 |
| 2 | 75 | 1,233 | 16 |
| 3 | 7 | 1,023 | 16 |

The profile captured 64 expert ranges and four MoE-layer ranges. Expert calls
can exceed the number of microbatches because the current implementation
submits separate calls for top-1 and top-2 selections.

## Measured ranges

Aggregated profiler ranges across the nine microbatches were:

| Range | CUDA time |
|---|---:|
| MoE layer 0 | 237.8 ms |
| MoE layer 1 | 243.5 ms |
| MoE layer 2 | 241.6 ms |
| MoE layer 3 | 239.2 ms |
| Attention layers 0–3 | 17.6, 16.5, 16.6, 16.2 ms |

The ranges are nested and therefore not additive with one another or with
backward operations. PyTorch also emits its generic profiler-cycle warning;
this probe does not call `prof.step()` between microbatches, and all nine
microbatches are present in the report. The range totals are retained as
attribution evidence, not as an independent speed claim.

## CUDA operator evidence

The largest self-device-time entries included:

- `aten::mm`: approximately 193.2 ms;
- CUTLASS BF16 GEMM kernels: approximately 67.0 ms and 35.0 ms for the two
  largest named kernel groups;
- `aten::copy_`: approximately 60.7 ms;
- `aten::nonzero`: approximately 52.7 ms;
- `aten::add_`: approximately 43.7 ms;
- an elementwise add kernel: approximately 40.1 ms;
- `aten::_index_put_impl_`: approximately 31.3 ms.

Overlapping indicative category totals were approximately 375.6 ms for
matrix/linear operations, 68.6 ms for routing/indexing, 119.2 ms for
layout/copy and 176.1 ms for activation/elementwise operations. These totals
overlap and must not be added.

## Decision

The first GPU Neural SSRE gate passes as an attribution/integrity experiment.
The concrete next step is a deterministic grouped-expert baseline that keeps
routing decisions, weights, expert parameters and outputs equivalent within
the established tolerance. It should measure whether grouping token subsets
reduces `nonzero`, `index_put`, copy/layout and small-launch overhead before
any neural scheduler is considered.

No neural scheduler has been trained or evaluated. No frozen Crucible
artifact, checkpoint or manifest was modified.

