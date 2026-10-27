# Crucible Phase VII: Learned Shared-Latent Relational Transfer

Date: 2026-10-27  
Status: preregistered design; no Phase VII training launched  
Branch: `research/two-candidate-forensics-20261027`

## Research question

Can a neural model learn a reusable relational representation from edge-offset
examples, rather than memorising the observed identifier pairs? Does an
auxiliary cycle-consistency objective improve transfer?

This phase is the first neural test of the Phase VI constructive idea. It is
not evidence that the original Crucible scorer discovered the representation;
that question remains a separate emergent-representation comparison.

## Frozen reference and conditions

The frozen reference is the accepted `SHARED_SCORER` dense-control family from
the topology-duration extension. Its final 2,000-update checkpoints are used
read-only as the N0 reference on the identical records and evaluation rows.
The reference has 22,106,906 parameters and both seeds passed the final
training gate on both graph families.

New neural conditions:

| Arm | Representation | Objective | Interpretation |
|---|---|---|---|
| N0 | Existing dense byte-sequence scorer | Candidate-slot cross-entropy | Frozen neural reference |
| N1 | Shared graph latent states | Candidate-slot cross-entropy | Constructive learned relational baseline |
| N2 | Same shared graph latent states | Candidate loss plus preregistered cycle loss | Test whether consistency supervision improves transfer |

N1 and N2 must use the same generated records, seeds, optimizer budget,
batch size and evaluation rows as N0. They may receive a canonical parsed
edge tensor derived from the same input record, but must not receive latent
potentials, held-out labels or a privileged answer feature. Parameter counts
must be reported; the target is within ±5% of N0 where feasible. If that
cannot be achieved, the comparison is labelled constructive and not
parameter-matched.

## N1/N2 architecture contract

The structured models operate on the observed graph edges and their modular
offsets, the query source and target markers, and the candidate values. They
do not receive the latent node values used by the data generator.

Each node starts from a shared, identifier-independent state with query-source
and query-target marker channels. A shared message/update block is applied to
all observed edges for a fixed number of rounds. The same relation head is
applied to every ordered node pair. The query prediction is formed from a
shared function of the target/source latent states (for example, a learned
difference representation), then mapped to the candidate slots.

N2 adds a cycle-consistency loss over a fixed, documented cycle basis of each
graph. Predicted modular edge distributions are composed around each cycle by
circular convolution; the target is the zero offset. The loss uses only
observed edge offsets and does not expose the held-out query result. The cycle
loss weight is fixed before training and is not tuned on validation outcomes.

The implementation must preserve a record of the parsed fields and verify
that no field is derived from `metadata.potentials`, `target_value`,
`target_slot`, or validation data.

## Data and splits

Primary Phase VII-A uses the immutable topology-control data:

- `P12_PRISM` and `R12_DIAMETER5`;
- 144 training rows and 96 validation rows per graph;
- both query directions and both candidate orders;
- the same eight held-out unordered node pairs;
- seeds 42 and 1337;
- 2,000 updates, cyclic schedule, batch size 12;
- no regenerated rows, changed modulus, symbolic assistance or sealed-test access.

Report these generalisation axes separately:

1. **Training relations:** observed training query pairs.
2. **Known-identifier pair transfer:** the fixed eight held-out pairs of the
   same identifiers.
3. **Order transfer:** candidate-order reversal and reversed query direction.
4. **Distance/topology:** per-pair and per-distance results for each graph.
5. **Identifier-renaming transfer:** a separately generated, prevalidated
   bijective relabelling with no learned identifier table allowed in N1/N2.
6. **New-identifier transfer:** a follow-up episode set whose identifiers do
   not occur in training. This is not pooled with the primary result and is
   only interpretable after the parser/oracle checks pass.

The first four axes are the primary Phase VII-A gate. Identifier-renaming and
new-identifier results are reported as separate secondary evaluations because
an embedding table over fixed node IDs cannot generalise to an unseen ID by
itself.

## Leakage and controls

- Generate the structured input from the serialized record, not from oracle
  metadata fields.
- Permit observed edge offsets because they are the task input.
- Prohibit latent potentials, target values, target slots, held-out outputs,
  validation rows and symbolic path answers from model inputs or persistent
  state.
- Verify the parser against an independent oracle and hash every split before
  training.
- Include a random-lookup baseline and the deterministic symbolic oracle as
  non-neural controls.
- Include candidate-order reversal, query-direction reversal, graph
  relabelling and a deliberately corrupted-input negative control.
- Keep N1 and N2 initialisation, data order, optimizer and compute budgets
  matched; use the same seed labels but independent model-specific parameter
  initialisation offsets recorded in the manifest.
- Do not use validation results for architecture, width, depth, loss-weight or
  checkpoint selection.

## Competence and success criteria

An arm is training-competent only if both seeds and both graph families reach
at least 95% row-level training value/slot accuracy by 2,000 updates. A
non-competent arm has no interpretable held-out transfer result.

Primary comparison:

- report paired held-out correctness, not only individual row accuracy;
- compare N1 and N2 with the frozen N0 reference on the same counterfactual
  groups;
- call a constructive transfer improvement only if the pooled held-out paired
  accuracy improves by at least 10 percentage points over N0, the direction is
  consistent across both seeds, and the paired-group confidence interval does
  not include zero improvement;
- a result that passes training but not this transfer criterion is a learned
  representation that did not demonstrate the intended generalisation.

Cycle-consistency claim:

- N2 supports an improvement over N1 only if it reaches the same competence
  gate, improves pooled paired transfer by at least 5 percentage points, and
  does not reduce training competence by more than 2 percentage points;
- otherwise report that the cycle objective did not demonstrate a benefit.

These thresholds are decision rules for this benchmark, not claims of a
universal reasoning threshold.

## Expected computational cost

The completed four-arm, 2,000-update dense duration extension took about 2 h
18 min wall time on the local CPU and produced about 339 MB of checkpoint and
trace artifacts. A conservative Phase VII-A budget is:

- four new N1/N2 arms (two graph families × two seeds): approximately one
  duration-extension equivalent if their backbone cost is comparable, plus
  cycle-loss overhead;
- reserve roughly 3.5–5 CPU hours and 0.5–1 GB for checkpoints, traces and
  predictions;
- run a 50-step throughput and memory preflight before committing to the full
  budget; stop if the model exceeds the recorded context or memory limits.

No cloud GPU is required by the protocol. A GPU may reduce wall time, but
compute accounting must report device, peak memory, updates, tokens/examples,
and wall time for every arm.

## Required artifacts and stop rules

Before execution, produce a parser/leakage audit, parameter-count report,
cycle-basis manifest, throughput preflight and final novelty audit. During
execution, produce per-arm predictions, training curves, checkpoint hashes,
metric JSON, and a final report separating N0/N1/N2 competence from transfer.

Stop before interpreting transfer if any structured arm fails parser/oracle
verification, if any arm receives a forbidden field, or if either primitive
training competence gate is not met. Do not add a new loss, change the graph,
increase the budget or tune the cycle weight after seeing validation results.
