# Cognitive manifest shortcut audit

Date: 2026-10-21
Scope: CPU-only screening before the instrumentation pilot

## Finding

The first manifest set passed its oracle and leakage checks, but its
error-correction labels had an unintended shortcut. In v1, candidate status
used `index % 2` while arithmetic composition also depended on the index. On
the two-composition validation split, prompt length therefore revealed the
label strongly enough for a simple length-bucket classifier to reach 75%.

This is a task-construction defect, not a model result. The v1 manifests and
their validation report remain preserved for provenance, but they are not
approved for GPU training.

## Corrected v2 result

Version 2 assigns candidate status using both composition slot and composition
block, balancing status within each composition while retaining deterministic
generation.

| Family | v1 validation length baseline | v2 validation length baseline | v2 majority baseline |
|---|---:|---:|---:|
| Error correction | 75.00% | 56.25% | 50.00% |
| Relational binding | 50.00% | 50.00% | 50.00% |
| Logic | 34.375% | 34.375% | 34.375% |
| Math | 0.00% | 0.00% | 0.00% |

The v2 error-correction baseline is not exactly chance because the screening
set is small and numeric surface lengths vary. It no longer exposes the
composition/label confound found in v1. The pilot should still report the
baseline and treat any future high shortcut score as a stop condition.

## Independent checks

The v2 validator reports:

- 640/640 serialized-oracle checks passed;
- zero cross-split input leakage;
- arithmetic composition holdouts preserved;
- relational path lengths 2, 3 and 4 preserved across train/validation/test;
- logical depths 2, 3 and 4 preserved across train/validation/test.

Machine-readable outputs:

- `experiments/crucible_cognitive_eval_v1_20261021/manifests/shortcut_audit.json`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/shortcut_audit.json`
- `experiments/crucible_cognitive_eval_v2_20261021/manifests/validation_report.json`

No GPU training, checkpoint download or cloud rental was used for this audit.
