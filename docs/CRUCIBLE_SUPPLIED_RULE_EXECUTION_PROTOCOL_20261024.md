# Crucible Supplied-Rule Execution Gate

Date: 2026-10-24  
Source benchmark: `crucible_rule_induction_benchmark_20261024_v3`  
Status: frozen before training

## Objective

Separate rule execution from rule induction. The model receives an explicit rule family, formula and parameter values, plus a query input. It must calculate the output for a rule instance that may be unseen during training.

## Arms

- Tiny execution control: 8 explicit-rule training tasks, used only to verify that the target path and execution interface can be fitted.
- Full static execution: 218 train rule instances and 63 validation rule instances from benchmark v3, with rule definitions supplied in every prompt.

The dense architecture, optimizer, tokenizer and raw-byte answer interface match the preceding CPU cognitive experiments. Seeds are `42` and `1337`; each arm runs exactly 2,000 updates with metrics at steps 0, 500, 1,000, 1,500 and 2,000. No temporary adapter or parameter update occurs at evaluation time.

## Rule presentation

The prompt explicitly supplies the family formula and its parameter values:

- affine2: `y=(a*x+b*z+c) mod10`;
- max-shift: `y=(max(x,z)+c) mod10`;
- absolute-difference-shift: `y=(abs(x-z)+c) mod10`;
- threshold-select: threshold and low/high offsets are supplied.

The hidden rule identifier is not included. Validation and test rule instances remain disjoint from training.

## Gates

- Tiny execution: final training exact accuracy at least 99%.
- Full validation execution: aggregate final exact accuracy at least 90% across both seeds.
- Test rule instances are evaluated once only if the full validation gate passes. Otherwise they remain sealed.

This gate does not test rule induction or meta-learning. Failure directs investigation toward representation, optimization or explicit arithmetic execution before adding adaptation loops.
