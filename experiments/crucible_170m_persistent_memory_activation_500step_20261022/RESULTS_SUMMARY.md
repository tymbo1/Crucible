# Crucible 170M persistent-memory activation test

Status: completed successfully on the local RTX 3060 Ti.

This was the bounded follow-up to the 100-update integration smoke test. It
used the unchanged 170,233,600-parameter E16 model, frozen context-128 data,
seeds 42 and 1337, all four factorial arms, and ten continuation phases of 50
attempted updates (500 attempted updates per arm and seed).

The skip criteria were not relaxed:

- maturity: 8 observations per example;
- stability tolerance: 0.02 absolute loss difference between the latest loss
  and the EMA.

## Final validation results

Lower bits per raw byte is better.

| Arm | Mean validation BPB | Mean raw-byte NLL | Applied/seed | Skipped/seed |
|---|---:|---:|---|---|
| baseline | 1.751024 | 1.213717 | 50, 50 | 0, 0 |
| memory-only | 1.749332 | 1.212545 | 50, 50 | 0, 0 |
| skip-only | 1.760124 | 1.220025 | 50, 50 | 0, 0 |
| full | 1.747097 | 1.210995 | 50, 50 | 0, 0 |

Memory was exercised in the memory-only and full arms. At phase 10, the loss
history reported 1,604 mature examples for seed 42 and 1,609 for seed 1337.
Nevertheless, no complete scheduled group satisfied the unchanged stability
criterion, so no update was skipped in either skip-enabled arm.

The memory arms recorded a 1.0 teacher hit rate after the first phase. This
confirms persistent scratchpad reuse, but the short run is not evidence that
the scratchpad improves the 170M model scientifically; the differences are
small and the skip intervention was inactive.

## Execution and provenance

- 80 phase records; all four arms × two seeds completed.
- Elapsed time: 10,939.08 seconds (3h 02m 19s).
- No CUDA or out-of-memory failure.
- No cloud or paid compute.
- The previous 100-update artifacts were not modified.

Interpretation: the activation/integration gate passed, eligibility matured,
and the detector correctly refused to skip without stable evidence. The
branch is now frozen pending return to C1r.
