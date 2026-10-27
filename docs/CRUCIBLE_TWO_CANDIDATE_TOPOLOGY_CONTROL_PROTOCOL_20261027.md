# Crucible Relational Topology Control v1

Date: 2026-10-27  
Status: frozen before execution  
Branch: `research/two-candidate-forensics-20261027`

## Objective

Test whether graph topology changes held-out relational-composition transfer
when identifier exposure, degree, edge count, training budget and optimizer
recipe are matched.

This is a new task family, deliberately separated from the earlier direct
identifier-matching experiments. The earlier pair panels only described which
pairs were shown; they did not require a relation to be composed along a path.

## Graph feasibility control

Both graph families contain 12 identifiers (`0`–`11`), 18 undirected edges,
and degree three at every identifier. Both are connected. The fixed families
are:

- `P12_PRISM`: the 6-cycle prism, diameter 4;
- `R12_DIAMETER5`: a fixed connected 3-regular graph, diameter 5.

The exact edge lists, degree sequences, components and pairwise distances are
generated and frozen by the preflight manifest. This satisfies equal vertex
exposure and equal edge count; topology is the intended remaining graph-level
factor, while the different distance distributions remain a measured property
rather than an assumed one.

## Relational rule and inference validity

Each episode assigns a hidden value `v(node)` in `Z/5Z`. Every graph edge
`u--v` is presented with the observable offset

`delta(u,v) = v(v) - v(u) mod 5`.

The query asks for the offset from a source node to a target node. The answer
is therefore recoverable by summing edge offsets along any path. Because edge
offsets are generated from node potentials, all paths are oracle-verified to
agree; this is an explicitly compositional relation rather than an arbitrary
graph label.

The prompt contains graph facts and the query, but never the hidden node
potentials or the answer. Five answer candidates (`0`–`4`) are presented in
two orders. The model must select the candidate whose value equals the
composed offset. The deterministic oracle recomputes the target independently.

## Fixed held-out pairs

The same eight unordered query pairs are held out in both graph families:

`(0,3), (0,4), (0,7), (0,8), (0,9), (0,10), (0,11), (1,3)`.

They are non-edges in both graphs. Both directions are evaluated, and each
semantic query is presented with both candidate orders. The preflight records
the shortest-path distance for every held-out pair in each graph. The held-out
pair identities are fixed; their distances are allowed to differ because that
is the topology contrast being measured.

## Data and training controls

Each condition has 72 semantic training episodes (36 fixed unordered query
pairs outside the held-out set, both directions), expanded to 144 rows by two
candidate orders. Each condition has 48 semantic validation episodes (the 8
held-out pairs, both directions, three independent hidden-potential draws),
expanded to 96 rows by candidate order.

The graph facts are deterministically shuffled per episode. Hidden potentials,
query directions and desired offsets are generated from a fixed data seed and
are identical in schedule across graph conditions. Training and validation
query-pair sets are disjoint. The model, dense `SHARED_SCORER` readout,
token-byte serialization, cyclic schedule, batch size 12, 500 updates,
optimizer, clipping and seeds 42/1337 remain unchanged from the accepted
two-candidate training path.

The primary competence gate is 95% row accuracy on the 144 training rows,
with no abstention target. Validation is interpreted only for arms that pass
this gate. Report both answer-value accuracy and candidate-slot accuracy,
because candidate order is deliberately varied.

## Required measurements

- training and validation row accuracy with denominators;
- target-value and selected-slot accuracy;
- candidate-order consistency and semantic counterfactual correctness;
- validation accuracy by fixed held-out pair, query direction and graph distance;
- graph degree, edge count, components, diameter and all held-out distances;
- per-seed variation and stepwise validation traces;
- oracle path-sum verification and a candidate-identity shortcut audit.

No claim of general graph reasoning is permitted from this benchmark alone.
Success would establish topology sensitivity or composition transfer under this
synthetic edge-offset task, not a neural mechanism or broad reasoning ability.

## Stop conditions

Stop before training if graph degrees/edge counts differ, either graph is
disconnected, a held-out pair is an edge, path sums are inconsistent, training
and validation query pairs overlap, the oracle target is exposed in the input,
candidate spans fail, token context overflows, class/order counts differ, or
the source/task implementation cannot reproduce the preflight manifest.

Do not access sealed tests, modify prior checkpoints, add auxiliary losses,
change the architecture, or silently rebalance a failed invariant.
