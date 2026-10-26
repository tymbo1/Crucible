# Checkpoint-Only Digitwise Operand Forensics

Date: 2026-10-25  
Status: frozen diagnostic  
Hardware: CPU only; no training

## Objective

Use existing single-digit addition checkpoints to characterize operand-interaction errors without changing training, data splits or model state.

## Scope

Evaluate all 55 valid ordered pairs `(a,b)` with `a+b<10` using the original canonical and factorized interfaces. The complete-pair matrix is exploratory. Original train/validation membership is preserved and reported separately.

## Measurements

- Complete operand-pair prediction matrix.
- Accuracy by operand order, sum magnitude, first operand and second operand.
- Accuracy on original train pairs, original validation pairs and remaining exploratory pairs.
- Commutativity: both directions correct, only `a,b`, only `b,a`, neither, and directional accuracy.
- Error type: predicted digit, modular error from the target and nearby/off-target predictions.
- Agreement and correctness consistency across available seeds/checkpoints.

## Safeguards

- Load checkpoints read-only; no optimizer or model updates.
- Do not evaluate any original sealed test partition.
- Mark complete-pair probes outside the original split as exploratory.
- Report unavailable repair-only arms explicitly; do not fabricate checkpoint results.
