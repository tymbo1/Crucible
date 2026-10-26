# Crucible Research Consolidation and Cardinality Audit

Date: 2026-10-26  
Scope: repository evidence consolidation and read-only audit of
`crucible_shared_candidate_scoring_cardinality_20261026_v1`.

No new training was run. No frozen scientific artifact was rewritten. No
sealed test set was accessed. The new machine-readable audit is
`experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/IMPLEMENTATION_AUDIT.json`.

## Ledger completion

At the original consolidation point, the coverage audit reported 27/27 final
experiment reports indexed and zero unindexed reports. After the competence
diagnostic below, the rerun reports 28/28 indexed reports and zero unindexed
reports. The ledger validator reports 28 records, no missing
artifact links and `valid: true`.

The 23 newly extracted reports preserve their exact accepted evidence through
the linked `REPORT_FINAL.md`, protocol, summary/analysis and finalization
artifacts. Statuses apply only to the stated hypothesis under the documented
conditions; `FAILED` does not mean that a broader architecture is impossible.

The full mechanism matrix is in
`docs/CRUCIBLE_MECHANISM_COVERAGE_20261026.json`. The principal overlaps are:

- The authority-selection lineage contains the v1/v2 role-binding runs,
  counterfactual diagnostics, representation/use audit, D1 causal-access
  experiment, and the already-indexed S4/component runs. The v2 raw-label
  counterfactual is the accepted clean result; the fixed-width run is
  exploratory because of its padding confound.
- The positional lineage contains the marker, position matrix, adjacency,
  position-generalization, paired-exposure and batch-order runs. They reuse
  the same A0/A1/A2/A3 family and are not independent architectural tests.
- The digitwise lineage contains balanced holdouts, diagonal ablations,
  role-swapped control, equality pretraining and variable-binding runs.
  Their pooled rows are correlated and the ablations retain stated marginal
  frequency limitations.
- The cardinality lineage contains the dynamic shared-scorer transfer and
  explicit-ID follow-up. The follow-up is a training-competence failure, not a
  clean transfer comparison.

## Cardinality implementation audit

### What is verified

| Check | Evidence | Finding |
|---|---|---|
| Task solvability | Deterministic identifier oracle | 2,064/2,064 rows correct |
| Data uniqueness | Per-interface JSONL audit | Train 576/576 unique IDs and inputs; validation 312/312; transfer 144/144 |
| Exact split overlap | Record IDs and exact input strings | Zero train/validation/transfer overlap within each interface |
| Candidate spans | Same deterministic regex path used by the runner | All candidate IDs, authority IDs and decision positions parsed; no span failures |
| Causal order | Position audit | All candidate and authority positions precede the decision position |
| Context bounds | Serialized input audit | Canonical/factorized lengths range from 56 to 167/179; no overflow |
| Label construction | Metadata target versus identifier-match rule | No label mismatches detected |
| Gradient path | Backward-only check on all eight checkpoints | Decision readout and 54/55 backbone parameters received gradients in every arm |
| Sealed-test boundary | Runner manifest and audit | `sealed_test_accessed: false` |

The evaluator and generated task therefore have a valid positive control. No
confirmed label, split-contamination, span-extraction, causal-visibility or
dead-decision-head defect explains the failed competence gate.

### What the learned run actually tests

The implementation does not train a neural candidate-span extractor.
`prepare_records()` locates `id=` spans with a deterministic regular
expression, and `batchify()` supplies candidate positions and masks to the
readout. The shared scorer receives those candidate positions directly. The
fixed-slot head receives a deterministic valid-slot mask. Consequently, the
run tests learned decision/readout behavior over supplied spans and a learned
backbone representation; it does not test candidate extraction as a learned
capability.

The backbone is trainable in both learned variants. `optimizer_for(model)` is
given the complete model, so the comparison is not a frozen-backbone readout
ablation. The architecture changes only the decision readout, but both the
backbone and readout are optimized from the arm's initialization.

Two implementation limitations are confirmed:

1. The inherited parser heads and the backbone language-model head are present
   in the model but are not used by the cardinality loss. The backward audit
   found these parameters without gradients in all eight arms. They are dead
   parameters for this objective, not evidence that the active decision path
   is disconnected.
2. The schedule is deterministic and cyclic. It presents 6,000 examples over
   500 batches of 12; 336 records are presented 10 times and 240 are presented
   11 times. Every batch has four target-slot-0 rows, four last-slot rows and
   four abstention rows. Within each cycle, 12 batches are two-candidate
   records followed by 36 batches of three-candidate records. There is no
   shuffle or interleaving of cardinalities.

### Coverage limitation

Training, validation and transfer authorities are always the first candidate,
the last candidate, or absent identifier `9`. In three-candidate examples,
slot 1 never receives a positive target; in transfer, interior candidate slots
also never receive positive targets. The experiment therefore does not test a
full candidate-ID matching function over all candidate positions. It tests
first/last matching and abstention under changing candidate counts.

This is a protocol coverage limitation, not a label error. It reduces the
strength of any claim about candidate-independent matching.

## Training-competence diagnosis

None of the eight learned arms crossed the preregistered 95% training gate:

| Variant | Training pointer accuracy across interfaces/seeds |
|---|---:|
| Fixed-slot | 384/576, 384/576, 384/576 and 388/576 |
| Shared scorer | 497/576, 444/576, 481/576 and 449/576 |

The traces contain 500 rows per arm with nonzero gradients and declining
current-batch loss. The final full-training accuracy remains below the gate,
so this is a genuine failure to master the recorded training distribution
under the accepted budget. It is not a clean held-out generalization result.

The gradient audit does not support a simple “no gradient” diagnosis. It does
support an optimization/representation bottleneck as still open: the current
loss is evaluated on the scheduled batch rather than a full-training snapshot,
the schedule is highly structured, gradient norms are clipped at 1.0 and
pre-clipping norms reach approximately 22.7--285.4 across the traces. These
facts make training dynamics plausible contributors, but do not identify a
single cause.

The existing results therefore support this narrower statement:

> The explicit-ID cardinality implementation is solvable and has an active
> gradient path, but its learned backbone/readout systems did not acquire the
> training task under the fixed 500-step cyclic protocol. Candidate-count
> transfer, shared-scoring superiority and learned candidate extraction remain
> untested or uninterpretable.

## Decision report

| Question | Status | Evidence-supported answer |
|---|---|---|
| Is the task/evaluator valid? | ESTABLISHED | Yes; oracle 2,064/2,064 and all static audits pass. |
| Did a learned candidate extractor fail? | NOT TESTED | No learned extractor exists in this implementation; spans are supplied by regex. |
| Did a learned matcher fail to master training? | ESTABLISHED | All eight learned arms are below 95% training pointer accuracy. |
| Is the failure caused by zero gradient? | REJECTED AS SOLE CAUSE | Active readout/backbone gradients are present in all eight backward checks. |
| Is an optimization bottleneck plausible? | SUPPORTED BUT UNRESOLVED | Structured cyclic batches, clipping and nonzero but variable traces support the hypothesis; no causal comparison was run. |
| Does shared scoring transfer across cardinality? | UNTESTED/UNINTERPRETABLE | Competence was not established, and interior slots were never positive targets. |
| Is an architecture-level ceiling established? | NO | The controls do not justify that conclusion. |

## Recommended next action

Do not launch another eight-arm cardinality-transfer run yet. If a follow-up
is approved, the highest-information low-compute diagnostic is a small,
training-competence-only factorial using the existing train records:

1. Keep the deterministic span extraction, labels, evaluator and candidate
   positions fixed.
2. Compare the current dense hidden representation with an explicit structured
   candidate-ID/query representation, so matching can be tested without asking
   the backbone to discover the representation.
3. Cross each representation with the current cyclic schedule versus a
   deterministic shuffled/interleaved schedule.
4. Use one interface and one seed first, require the 95% training gate, and
   inspect full-training accuracy at fixed checkpoints. Do not evaluate
   cardinality transfer until the gate passes.

This four-cell diagnostic distinguishes representation failure from schedule
or optimization failure at far lower cost than another eight-arm transfer
study. It must be preregistered and subjected to the ledger novelty check
before execution. The present report does not run it.

## Follow-up competence diagnostic

The preregistered four-cell diagnostic was subsequently run after the novelty
audit. It evaluated only the canonical 576-row training split:

| Cell | Training pointer accuracy | Gate |
|---|---:|---|
| Dense/cyclic | 497/576 | FAIL |
| Dense/shuffled | 384/576 | FAIL |
| Structured/cyclic | 390/576 | FAIL |
| Structured/shuffled | 432/576 | FAIL |

The dense/cyclic cell reproduced the prior result. Schedule changes altered
competence within both representations, but no cell passed and structured
features did not rescue training. The structured model was not
parameter-matched and never selected the 48 slot-1 targets, so the result is
inconclusive about representation quality. Cardinality transfer remains
closed. See the [diagnostic report](../experiments/crucible_cardinality_competence_diagnostic_20261026_v1/REPORT_FINAL.md)
and its [ledger entry](CRUCIBLE_RESEARCH_LEDGER.json).

## Audit artifacts

- [Machine-readable implementation audit](../experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/IMPLEMENTATION_AUDIT.json)
- [Audit script](../tools/audit_crucible_shared_candidate_scoring_cardinality_20261026.py)
- [Accepted cardinality report](../experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1/REPORT_FINAL.md)
- [Cardinality protocol](CRUCIBLE_SHARED_CANDIDATE_SCORING_CARDINALITY_PROTOCOL_20261026.md)
- [Complete mechanism coverage](CRUCIBLE_MECHANISM_COVERAGE_20261026.json)
- [Research ledger](CRUCIBLE_RESEARCH_LEDGER.json)
