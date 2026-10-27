# Crucible Topology Control: Training-Duration Extension v1

Date: 2026-10-27  
Status: frozen before execution  
Branch: `research/two-candidate-forensics-20261027`

## Objective

Determine whether the three topology-control training-gate failures reflect
slower acquisition or a genuine plateau under the frozen task and model.

This is a competence-rescue diagnostic, not a new topology comparison. It
repeats the exact four topology-control arms from
`crucible_two_candidate_topology_control_20261027_v1` from the same random
initialization and data schedule, but extends training from 500 to 2,000
updates. The first 500 updates must reproduce the committed baseline before
the extension is interpreted.

## Fixed inputs and controls

- Immutable training and validation JSONL from the accepted topology-control
  experiment; no regenerated splits or new graph data.
- `P12_PRISM` and `R12_DIAMETER5`, each 12 nodes, 18 edges, degree three and
  connected.
- Same eight held-out unordered node pairs, both query directions and both
  candidate orders.
- Same 22M dense-control architecture, tokenizer, `SHARED_SCORER` readout,
  optimizer, clipping, batch size 12, cyclic ordering and seeds 42/1337.
- Same initialization and first-500-update schedule; no checkpoint resume,
  because the original checkpoints did not preserve optimizer state.
- No architecture changes, auxiliary losses, symbolic assistance or sealed
  test access.

## Duration and measurements

Train each arm for 2,000 total updates. Record training and validation
metrics at updates `1, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500,
1750, 2000`.

The 95% row-level training gate remains unchanged. Report:

- final training and validation value/slot accuracy with denominators;
- paired candidate-order correctness and candidate-order flips;
- first update at which the training gate is reached, if reached;
- training accuracy trajectories and final error rows by graph condition;
- per-seed variation and comparison with the committed 500-step results.

Validation is interpreted only for arms that pass the final 2,000-update
training gate. Earlier validation values are trajectory diagnostics, not
generalization evidence.

## Decision rules

- If the diameter-5 arms reach the gate later, the original result is a
  training-speed/optimization finding; topology transfer remains untested.
- If all arms reach the gate, a later topology comparison must use the same
  2,000-update budget across conditions.
- If the diameter-5 arms remain below the gate while the prism arms pass, the
  task has a graph-condition-dependent acquisition failure under this recipe;
  do not attribute it to path length alone.
- If all arms plateau below the gate, inspect representation and optimizer
  dynamics before claiming anything about relational composition.

This extension cannot establish a graph-distance effect by itself. It only
determines whether the prior topology comparison was blocked by insufficient
training competence.
