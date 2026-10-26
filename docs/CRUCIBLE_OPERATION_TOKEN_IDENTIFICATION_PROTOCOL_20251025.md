# Crucible Operation-Token Identification Control

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only; no sealed-test access

## Objective

Test whether the 22M dense control can recognize an explicit operation identifier independently of executing the requested operand-selection behavior.

## Task

Each example contains two operands and either `operation=SELECT_FIRST` or `operation=SELECT_SECOND`. The target is a class digit:

- `0` for `SELECT_FIRST`;
- `1` for `SELECT_SECOND`.

The target is therefore an operation label, not an operand value.

The experiment reuses the frozen role-selection pair partition: 40 disjoint training pairs and 10 validation pairs, including equal and unequal pairs and reversed unequal-pair pairs. Both operation labels are present for every pair.

## Arms

The canonical and factorized interfaces are run with fresh seeds `42` and `1337`, unchanged dense architecture, tokenizer, optimizer, evaluator and 500-update budget. This produces four arms.

## Gates

- training exact accuracy ≥95%;
- overall validation exact accuracy ≥90%;
- output-token and EOS correctness are reported separately;
- deterministic class oracle, split audit and checkpoint hashes must pass.

No result is used to claim that the model can execute the selected procedure. It only tests recognition of the operation identifier under the frozen prompt format.
