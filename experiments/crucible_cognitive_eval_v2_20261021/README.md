# Crucible cognitive evaluation v2

This directory contains the corrected CPU-only task manifests for the first
instrumentation pilot. It is not a model-training result.

Version 1 remains preserved as an audited historical artifact. Its
error-correction generator correlated candidate status with arithmetic
composition on the validation split, so v1 must not be used for a GPU pilot.
Version 2 keeps the same task families and composition/depth holdouts while
balancing correction status within each arithmetic composition.

## Validation

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_cognitive_manifests.py \
  --manifests experiments/crucible_cognitive_eval_v2_20261021/manifests

PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_cognitive_shortcuts.py \
  --manifests experiments/crucible_cognitive_eval_v2_20261021/manifests
```

The independent validator checks all 640 exact oracles, cross-split input
leakage and composition/path/depth holdouts. The shortcut report records
target distributions and cheap majority and prompt-length baselines; these
are screening checks, not learnability evidence.

No training, cloud execution or checkpoint download is required.
