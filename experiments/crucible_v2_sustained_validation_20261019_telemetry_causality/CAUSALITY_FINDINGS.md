# Crucible V2 telemetry-causality findings

Date: 2026-10-19 (Australia/Brisbane)

## Outcome

The earlier 100-step equivalence failure is attributable to the GPU weighted
telemetry reductions in this frozen Crucible/PyTorch/RTX 3060 Ti setup.

The decisive test was the narrow replacement that moved only the two weighted
`torch.bincount` reductions to CPU:

1. weighted selected-probability accumulation; and
2. weighted selected-count accumulation.

With that replacement, both the original and grouped-rank-preserving engines
produced identical model, gradient, and optimizer-state SHA-256 values through
100 steps for seeds 42 and 1337. The broader all-CPU telemetry reference also
passed the same gate.

This isolates the causal path to the pair of weighted GPU reductions. It does
not yet distinguish which individual reduction is sufficient, and it does not
prove that every CUDA telemetry operation is deterministic on every platform.

## Evidence matrix

| Diagnostic | Steps | Repetitions | Result |
|---|---:|---:|---|
| CPU-reference telemetry | 10 | 1/variant | Passed |
| CPU-reference telemetry | 100 | 1/variant | Passed |
| Only weighted `bincount` reductions on CPU | 10 | 2/variant | Passed |
| Only weighted `bincount` reductions on CPU | 100 | 1/variant | Passed |

For every passing 100-step diagnostic, seeds 42 and 1337 matched across
original versus grouped execution for:

- terminal model state;
- terminal gradients; and
- terminal optimizer state.

The corrected diagnostics also emitted all four layer telemetry records
(`layer_0_moe` through `layer_3_moe`), so the pass was not caused by disabling
telemetry.

The earlier strict no-telemetry test also passed through 100 steps. In contrast,
the prior full GPU-telemetry test failed, with model/gradient/optimizer hashes
diverging and validation-NLL deltas of approximately 0.00077–0.00098.

## Interpretation

The grouped V2 execution path is not implicated by the corrected causality
tests. The most defensible production repair is to replace the two weighted
CUDA reductions with a deterministic implementation, initially the CPU
reference or another separately validated deterministic reduction.

The 3,125-step sustained V2 validation remains gated. It must not launch until
the production telemetry implementation is patched and a fresh 100-step gate
using that production path passes.

The approximately 37% short-benchmark execution reduction remains a measured
performance result, but it is not yet promoted to a sustained-training result.

## GPU health

At the final health check:

- `nvidia-smi` succeeded;
- GPU: RTX 3060 Ti;
- driver: 580.126.18;
- PyTorch reported `cuda=True`, one device, and BF16 support;
- kernel logs showed driver initialization only, with no observed XID, fallen-GPU,
  reset, or GPU-OOM record in the inspected range.

The earlier `nvidia-smi` communication failure was transient. No evidence from
this investigation supports rebooting or replacing the GPU, and no reboot was
performed.

## Integrity boundary

All diagnostics used the frozen configuration and anchors. The script installed
the telemetry replacements in memory only. No frozen checkpoint, manifest,
training run, RunPod artifact, or previous analysis was modified.

## Valid artifacts

Directory:

`/home/timbushnell/crucible_v2_sustained_validation_20261019_telemetry_causality/`

- `run_cpu_reference_telemetry.py`
- `cpu_reference_correct_10/cpu_reference_causality_report.json`
- `cpu_reference_correct_10/cpu_reference_hashes.json`
- `cpu_reference_correct_100/cpu_reference_causality_report.json`
- `cpu_reference_correct_100/cpu_reference_hashes.json`
- `weighted_bincount_cpu_correct_10/cpu_reference_causality_report.json`
- `weighted_bincount_cpu_correct_10/cpu_reference_hashes.json`
- `weighted_bincount_cpu_correct_100/cpu_reference_causality_report.json`
- `weighted_bincount_cpu_correct_100/cpu_reference_hashes.json`

SHA-256 values recorded at sealing:

| Artifact | SHA-256 |
|---|---|
| `run_cpu_reference_telemetry.py` | `cd51f17b6b0d696f0c642704b32f0f8827d304b725c00f9685990439466648b7` |
| CPU 10-step report | `fdbc87c4b51a05ae8b79142fd71b3261f81db16f2307a189ea6e6fbc8a9079c4` |
| CPU 10-step hashes | `b6a537aa2be84a227bfc39377b6c1efed6237b6c85bbb579cb38bb2811a44364` |
| CPU 100-step report | `1ec452c5fc355afaf2a9ce636eefadb474acdf576e44a1aab0c329fd599163bd` |
| CPU 100-step hashes | `25e7734343763f9d78c096e0a1daf2856f716b59e2089a8fb15836da19d0c62e` |
| Weighted-CPU 10-step report | `7962a57da4d32d809f51b9ef3bf450859c1493f6699fad05e06f7896ecfe5147` |
| Weighted-CPU 10-step hashes | `139c8eaf456a581c6a12c7a115c91afee7916eb8bb552670e0e67e38f6440c09` |
| Weighted-CPU 100-step report | `93332dadf6f78d130353dd19b79d981d61a08a85f7a6854af2474e0d946f4ff3` |
| Weighted-CPU 100-step hashes | `e9101da80179c21386f99b997ab5af040bd8084711a2839d6229d8b52d4bc5f8` |

