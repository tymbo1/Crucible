# Crucible D2 Component Ablation v1

Date: 2026-10-26  
Status: preregistered CPU ablation

## Objective

Identify which changes separated the failed original D2 configuration from
the successful S4 configuration:

1. candidate-slot pointer versus operation-class readout;
2. decision-only loss versus the original auxiliary multi-task loss.

The accepted S4 candidate-slot/decision-only checkpoints and accepted original
D2 operation-class/multi-task checkpoints are reused as diagonal controls.
Only the two crossed cells are newly trained.

## Frozen task and controls

- Use the exact standard S3 late-authority train/validation records from the
  accepted D2 readout ladder.
- Preserve canonical and factorized interfaces, seeds 42 and 1337, 500
  updates, effective batch size 12, optimizer and tokenizer.
- The candidate-slot target is the authoritative candidate index (0 or 1).
- The operation-class target is `SELECT_FIRST` or `SELECT_SECOND`.
- No sealed-test access.
- Accepted S4 and original D2 checkpoints are not modified.

## Factorial cells

| Cell | Readout | Objective | Source |
|---|---|---|---|
| C1 | Candidate slot | Decision only | Reused accepted S4 |
| C2 | Candidate slot | Auxiliary multi-task | Newly trained |
| C3 | Operation class | Decision only | Newly trained |
| C4 | Operation class | Original D2 multi-task | Reused accepted D2 |

The auxiliary multi-task objective matches the original D2 loss: decision loss
plus value classification for operands A/B and pointer losses for A/B. The
operation-class cell does not add an operation-position loss, matching D2.

## Primary metrics

- training pointer/selection accuracy;
- standard validation selection accuracy;
- paired counterfactual flips when only authority changes;
- paired two-way correctness;
- output correctness and format errors;
- seed and interface variation.

The competence gate is at least 95% training selection accuracy. Validation
results from arms below that gate are reported but not interpreted as clean
generalization evidence.

## Interpretation

- C1 success and C2 failure implicate the auxiliary objective or its
  interaction with the candidate-slot head.
- C1 success and C3 failure implicate operation-class readout/targeting.
- C2 and C3 success but C4 failure implicates the combination present in the
  original D2 setup.
- C1 and all other cells success would weaken both hypotheses and motivate a
  broader reproducibility check.

This experiment does not test three-way composition, arbitrary authority rules
or autonomous reasoning. Those are subsequent gates after the component cause
is better isolated.
