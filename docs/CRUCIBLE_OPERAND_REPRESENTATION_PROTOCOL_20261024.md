# Operand Representation and Exposure Control

Date: 2026-10-24  
Status: frozen before training  
Hardware: CPU only

## Objective

Test whether the single-primitive transfer failure persists when the operand is represented canonically versus factorized into independently visible fields, while extending exposure beyond the 500-update ladder.

The underlying rule is fixed throughout the substantive comparison:

`y = (x + 1) mod 10`

Operands are drawn from `0..99`. Eighty values are used for training and twenty disjoint values for validation; every tens and ones digit appears in training.

## Arms

| Arm | Input representation | Purpose |
|---|---|---|
| Canonical numeric | `x=47` | Baseline serialized numeric operand |
| Factorized | `operand_tens=4; operand_ones=7` | Separates operand fields while preserving numeric information |
| Opaque association | `operand_symbol=<opaque label>` | Negative control for memorized associations; the symbol-to-value mapping is hidden from the model |

The opaque arm is not expected to solve a task whose mapping is unavailable. It is included to show the behavior of an association-only interface and must not be treated as evidence against arithmetic execution.

## Neural controls

- Existing dense Stage-A control, unchanged architecture, tokenizer and optimizer.
- Seeds `42` and `1337`.
- 1,000 updates per arm and seed.
- Checkpoints at steps 0, 250, 500, 750 and 1,000.
- Same operand split, targets, data order policy and training budget for canonical and factorized arms.
- No GPU, E4 routing, original sealed test evaluation or cloud resources.

## Safeguards

- A deterministic symbolic executor computes every target.
- Preflight requires 100% oracle accuracy and zero train/validation prompt overlap.
- Every tens and ones digit occurs in the substantive training split.
- The validation operand IDs remain disjoint and are not used for checkpoint selection.
- Results are reported separately for training fit and unseen-operand transfer.

## Interpretation gates

- If canonical reaches high training accuracy but factorized transfer improves, representation is implicated.
- If both fit training data and both fail transfer, the failure is not repaired by simple operand factorization.
- If neither fits training data, exposure/optimization remains unresolved.
- Opaque-arm results are association-control evidence only.

No result authorizes GPU scaling, neuromorphic integration or a larger model without a separately approved protocol.
