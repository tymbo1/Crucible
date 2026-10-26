# Crucible Two-Candidate Pair-Coverage Experiment v1

Date: 2026-10-27
Status: frozen before execution
Branch: `research/two-candidate-forensics-20261027`

## Objective

Distinguish dependence on exact training candidate-pair coverage from a more
general identifier-matching rule. Rotate which unordered candidate pairs are
held out while preserving the dense architecture, optimizer, cyclic schedule,
update budget, target balance, candidate order and context/interface controls.

This is a new training experiment. It does not access sealed test data and it
does not modify prior checkpoints or prior experiment artifacts.

## Pair panels

Use four disjoint perfect matchings over identifiers 0–7:

- `M0`: `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`
- `M1`: `(0,2)`, `(1,3)`, `(4,6)`, `(5,7)`
- `M2`: `(0,3)`, `(1,2)`, `(4,7)`, `(5,6)`
- `M3`: `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`

Each unordered pair occurs in exactly one panel. The four partitions are:

| Partition | Training panel | Held-out validation panel |
|---|---|---|
| `P0` | `M0` | `M1` |
| `P1` | `M1` | `M2` |
| `P2` | `M2` | `M3` |
| `P3` | `M3` | `M0` |

Thus every panel is used as training once and held out once. A validation pair
is never present in that partition's training set, while every identifier is
present in training through the perfect matching.

## Context and serialization controls

Use the canonical source training contexts as the fixed context pool:

`(1,6)`, `(1,8)`, `(2,7)`, `(2,9)`, `(3,8)`, `(4,9)`.

Training uses all six contexts. Validation uses the fixed four-context subset
`(1,6)`, `(2,7)`, `(3,8)`, `(4,9)`, all of which are represented during
training. This deliberately removes the original unseen-context confound.

Serialize both training and validation with the canonical `TRAIN_COMPACT`
layout. Candidate pairs and record IDs remain disjoint within each partition;
inputs are regenerated deterministically from the frozen panel/context rule.

For each context, ordered pair and authority, include the two candidate
authorities and `9` abstention. Therefore every partition has:

- 144 training rows: 48 target slot 0, 48 target slot 1, 48 abstentions;
- 96 validation rows: 32 target slot 0, 32 target slot 1, 32 abstentions;
- both candidate orders for every pair;
- 48 training and 32 validation counterfactual groups.

## Training arms

Use the existing dense `SHARED_SCORER` model and accepted training path:

- Seeds: `42`, `1337`;
- schedule: deterministic cyclic only;
- effective batch size: `12`;
- updates: `500`;
- optimizer, learning rate, clipping and 22M architecture unchanged;
- training competence gate: at least `95%` row-level accuracy on all 144 rows.

Validation is evaluated for every arm after training, but conclusions about
transfer require the training gate to pass. Report training and validation
accuracy, slot-level accuracy, abstention, paired correctness, counterfactual
flips, pair-panel accuracy and source-context accuracy.

## Decision criteria

- If validation errors follow the held-out panel—different panels fail when
  withheld—this supports dependence on exact pair coverage.
- If the same unordered pairs or identifier families fail across partitions
  despite being trained in another partition, this supports a persistent
  identifier- or representation-specific effect.
- If all panels transfer similarly after competence is reached, the original
  failure was likely specific to the original split or training trajectory.
- If training competence fails in some partitions, report that separately and
  do not interpret their validation as transfer evidence.

These criteria distinguish behavioral explanations only. They do not identify
an internal neural mechanism.

## Stop conditions and invariants

Stop before training if panel disjointness, identifier coverage, class counts,
context overlap, oracle labels, candidate spans or deterministic input
construction fail. Do not repair an invariant by silently changing the model,
budget, tokenizer or evaluator.

No architecture change, memory module, auxiliary loss, curriculum, shuffled
schedule or sealed-test access is permitted in this experiment.
