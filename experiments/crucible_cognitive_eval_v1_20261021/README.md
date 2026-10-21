# Crucible cognitive evaluation v1

This is a preserved historical artifact, not the pilot input. The CPU
shortcut audit found that v1 error-correction candidate status was correlated
with arithmetic composition on validation, producing a 75% prompt-length
bucket baseline. Use the corrected v2 manifests in
`experiments/crucible_cognitive_eval_v2_20261021/` for any future training.

This directory contains CPU-only task definitions and deterministic manifests.
It is not a model-training result.

## Included task families

- exact integer arithmetic;
- directed relational binding with distractors and path-length holdouts;
- forward-chaining logic with entailed/contradicted/unknown labels;
- candidate-answer verification and correction.

The validation and test splits hold out operation compositions, relation path
lengths or rule depth. Every record includes a deterministic reference oracle;
the oracle must not be supplied as model input.

## Rebuild

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/build_cognitive_manifests.py \
  --output /tmp/crucible_cognitive_eval_v1_rebuild
```

The builder refuses to overwrite a non-empty output directory. Compare the
fresh hashes with the checked-in manifest; the checked-in manifest and JSONL
hashes are the immutable v1 instance.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_cognitive_manifests.py \
  --manifests experiments/crucible_cognitive_eval_v1_20261021/manifests
```

The validator independently parses the task text, checks all 640 exact
oracles, detects cross-split input leakage and verifies composition, path
length and rule-depth holdouts.

No training, cloud execution or checkpoint download is required.
