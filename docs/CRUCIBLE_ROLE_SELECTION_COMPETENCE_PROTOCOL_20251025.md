# Crucible Primitive Role-Selection Competence Gate

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only; no sealed-test access

## Objective

Determine whether the existing 22M dense-control model can learn operand selection as an isolated primitive before using it as evidence for variable-binding transfer.

The experiment separates fixed single-operation execution from operation-conditioned selection:

- `S1_SELECT_FIRST`: execute a fixed first-operand selection instruction;
- `S2_SELECT_SECOND`: execute a fixed second-operand selection instruction;
- `S3_JOINT_OPERATION_CONDITIONED`: select the requested role from an explicit operation identifier.

No addition task, persistent memory, recursive controller, symbolic execution assistance, or architecture change is included.

## Frozen data

The task uses disjoint operand-pair splits. Equal and unequal pairs are present in both splits; competence is scored primarily on unequal validation pairs because equal operands do not distinguish the selected role.

Validation pairs are frozen as:

`(0,0), (5,5), (0,9), (9,0), (1,8), (8,1), (2,7), (7,2), (3,6), (6,3)`.

The 40 training pairs are the previous role-binding training pairs with `(0,0)` and `(5,5)` removed and `(4,7)` and `(7,4)` added. The exact ordered pair lists are written to `PREFLIGHT.json` before training.

The validation set contains two equal and eight unequal pairs, including four reversed unequal-pair pairs. The experiment reports equal, unequal, reversed-pair, first-position and second-position coverage explicitly.

## Arms and interfaces

Each condition runs with both the existing `canonical` and `factorized` byte-serialized interfaces, using fresh seeds `42` and `1337` and the unchanged 22M dense model, tokenizer, optimizer, 500-update budget and evaluator.

| Condition | Prompt behavior | Purpose |
|---|---|---|
| `S1_SELECT_FIRST` | Fixed instruction: select the first operand | Isolate first-role execution |
| `S2_SELECT_SECOND` | Fixed instruction: select the second operand | Isolate second-role execution |
| `S3_JOINT_OPERATION_CONDITIONED` | Explicit `operation=SELECT_FIRST` or `SELECT_SECOND` | Test selecting which primitive to execute |

The final run therefore contains 12 arms. Every arm exports a checkpoint, prediction rows, curve, and hash.

## Gates

For each condition/interface/seed:

- training exact accuracy must be at least 95%;
- overall validation exact accuracy must be at least 90%;
- unequal-validation exact accuracy must be at least 90%.

The unequal-validation score is the primary competence measure. Equal-validation results remain diagnostic and are not used alone to claim role competence.

The experiment does not advance to conditional transfer unless all S1, S2, and S3 arms pass all gates.

## Stage 1 audit

Before training, audit the completed variable-binding experiment:

- record counts, pair and operation coverage;
- target and oracle consistency;
- output-token, EOS, and format correctness;
- whether predictions equal the first operand, second operand, either operand, or neither;
- operation-specific and split-specific accuracy;
- operand-order and reversed-pair coverage.

This audit is descriptive and does not modify the prior artifacts.

## Conditional transfer stop rule

If all competence gates pass, evaluate the joint S3 checkpoints on a separately frozen set of newly held-out ordered pairs and report this as `conditional_transfer`. If any individual primitive or S3 fails, do not run or interpret this stage; preserve the failure curves and predictions instead.

## Required artifacts

- this protocol;
- `PREVIOUS_FAILURE_AUDIT.json` and `.md`;
- preflight and immutable split records;
- per-arm prediction CSV and machine-readable metrics;
- per-arm training curves;
- deterministic oracle verification;
- checkpoint and data hashes;
- final report distinguishing primitive acquisition, operation-conditioned execution, and conditional transfer.

No result establishes general reasoning or autonomous self-learning. A failed gate is a capability-localization result, not an architectural conclusion.
