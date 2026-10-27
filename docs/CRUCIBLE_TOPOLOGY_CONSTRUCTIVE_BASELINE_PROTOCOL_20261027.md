# Crucible Topology Constructive Relational Baseline v1

Date: 2026-10-27  
Status: read-only symbolic control  
Branch: `research/two-candidate-forensics-20261027`

## Objective

Independently reproduce the Phase VI constructive baseline proposed in the
symbolic-space handoff. The experiment tests whether a shared latent-potential
representation is sufficient to compose unseen modular relationships on the
two committed topology graphs. It is an explicitly supplied symbolic
algorithm, not a neural-learning experiment.

## Fixed inputs

- Use the committed graph manifest from
  `experiments/crucible_two_candidate_topology_control_20261027_v1/GRAPH_MANIFEST.json`.
- Use the same eight held-out unordered node pairs as the topology-control
  task.
- Test `K ∈ {5, 11, 97}`.
- Run 1,000 random latent assignments per graph and modulus.
- Use a deterministic random seed recorded in the output.

## Conditions

For each graph and modulus:

1. Sample a latent value `z_i ∈ {0, …, K-1}` for every node.
2. Expose only edge offsets `(z_v - z_u) mod K` on the training edges.
3. Recover node potentials from one root by graph traversal and compose each
   held-out relation with `(z_b - z_a) mod K`.
4. Compare against an independent random lookup baseline, whose expected
   accuracy is `1/K`.
5. Corrupt one randomly selected training edge by `±1 mod K` and repeat the
   shortest-path composition. Report the resulting accuracy.
6. Check that the uncorrupted observations satisfy every edge constraint after
   potential recovery. This is an implementation and consistency check.

The implementation also records whether the corrupted observation is detected
as cycle-inconsistent by the recovered potentials. This is a property of the
symbolic representation, not a claim about Crucible's neural internals.

## Claims that may be supported

- A shared latent-potential representation is sufficient for exact transfer
  to unseen pairs when edge observations are consistent.
- The symbolic task is logically identifiable on both committed graphs for the
  tested moduli.
- A local corrupted edge can reduce shortest-path composition accuracy, with
  topology-dependent effects.

## Claims that are explicitly out of scope

- Crucible has learned the latent-potential representation.
- Crucible implements BFS, shortest paths, cycle checks or modular subtraction.
- The symbolic corruption accuracy predicts neural checkpoint errors.
- A per-node latent table can generalise to entirely unseen identifiers; this
  experiment only tests unseen pairs of identifiers present in the graph.
- The result establishes general relational reasoning or multi-hop learning.

## Acceptance and reproducibility

The run is accepted if:

- all source hashes and output hashes are recorded;
- every uncorrupted symbolic trial reconstructs all held-out relations;
- the random lookup baseline is close to its `1/K` expectation within the
  reported finite-sample variation;
- the report preserves the distinction between supplied symbolic inference and
  learned neural competence; and
- no model training or sealed-test access occurs.

The script must not read model checkpoints or alter any scientific artifact
from the neural topology runs.
