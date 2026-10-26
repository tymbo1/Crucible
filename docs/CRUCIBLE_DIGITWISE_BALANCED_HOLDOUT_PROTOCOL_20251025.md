# Digitwise Addition — Repeated Balanced Pair-Holdout Study

Date: 2026-10-25  
Status: frozen before training  
Hardware: CPU only

## Objective

Determine whether the four failures observed in the fixed 2026-10-25 ladder split—`(1,5)`, `(2,2)`, `(3,3)` and `(5,1)`—persist when the train/validation partition is rotated, or whether the failures move with sparse pair coverage.

This is a split-distribution study. It is not an architecture, optimizer, carry, recursion or neuromorphic experiment.

## Frozen controls

- Existing 22M dense model and initialization procedure.
- Existing byte-level tokenizer, `\nAnswer:` marker, target masking and evaluator.
- Existing direct-training recipe, AdamW settings and 500 updates.
- Interfaces: canonical and factorized.
- Initialization seeds: `42` and `1337`.
- 40 training pairs and 15 validation pairs per partition.
- No curriculum arm, because the earlier direct/curriculum checkpoints were prediction-identical within each interface/seed; direct is the controlled recipe here.
- No sealed test corpus access.
- No architecture, tokenizer, optimizer, update-budget or task-definition changes.

## Pair universe and partitions

The universe is the 55 ordered pairs satisfying `a,b∈{0,…,9}` and `a+b<10`. Every partition contains each pair exactly once, with a 40/15 train/validation split.

Four deterministic partitions are frozen in `PARTITIONS.json`:

| Partition | Purposeful validation inclusion |
|---|---|
| `balanced_a` | All four previously problematic pairs; diagonal `(2,2)` and `(3,3)` |
| `balanced_b` | None of the four problematic pairs; diagonal `(0,0)` and `(4,4)` |
| `balanced_c` | `(1,5)` and `(2,2)`; diagonal `(1,1)` and `(2,2)` |
| `balanced_d` | `(5,1)` and `(3,3)`; diagonal `(0,0)` and `(3,3)` |

Each partition has two diagonal/equal-operand validation pairs and thirteen non-diagonal validation pairs. The balance objective minimizes deviation from the full-universe frequencies of first operand, second operand and sum, while also controlling reversed-pair placement. Perfect Latin-square balance is not claimed because the valid domain is triangular: digit 9 appears in only two valid ordered pairs, while digit 0 appears in ten.

The four problematic pairs appear in validation in some partitions and training in others. The resulting split assignments are fixed before training and are never selected using new model outputs.

## Preflight checks

Before any model update, the runner must verify:

- exactly 55 unique valid pairs per partition;
- 40/15 train/validation counts;
- no pair overlap between train and validation;
- correct labels and oracle sums;
- one-digit targets and fixed interface formatting;
- two diagonal validation pairs per partition;
- complete operand/sum coverage diagnostics;
- all purposeful inclusion/exclusion constraints;
- no test path is read.

The runner writes `PARTITIONS.json` and `PREFLIGHT.json` before starting training.

## Measurements

For every partition, interface and seed, record:

- training and validation exact accuracy;
- training competence gate (`≥95%` exact training accuracy);
- every pair prediction and output token;
- split identity, partition identity and checkpoint hash;
- accuracy by pair, first operand, second operand, sum, absolute difference and operand order;
- diagonal versus matched non-diagonal validation accuracy;
- outcomes for the four previously problematic pairs;
- variation across independent seeds and partitions.

The evaluator's semantic output is one digit, while its target mask includes the digit and EOS. Both output-digit correctness and full evaluator sequence correctness are recorded.

## Interpretation gates

- If failures move with held-out pairs, sparse coverage/distribution is implicated.
- If the same pairs fail across materially different partitions, pair-specific representation or optimization effects become more plausible.
- If diagonal pairs underperform matched non-diagonal controls across partitions, a diagonal-specific follow-up is justified.
- If errors remain broad across partitions, the failure is a more general transfer limitation.

None of these outcomes establishes an embedding-collision or attention-interference mechanism. Internal activation analysis is a separate follow-up and is not part of this run.

## Stopping and disclosure

If any invariant fails, the runner must stop and report the failure rather than silently changing the architecture, budget, data or evaluator. Results are exploratory distribution-shift evidence, not a replacement for a final untouched benchmark.
