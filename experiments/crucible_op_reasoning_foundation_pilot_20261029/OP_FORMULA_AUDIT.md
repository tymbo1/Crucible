# OP recursive reasoning audit

Date: 2026-10-29

## Recovered definition

The exact OP specification is recorded in
[`docs/CRUCIBLE_RECURSIVE_CONTROLLER_OPERATIONAL_SPEC_20261021.md`](../../docs/CRUCIBLE_RECURSIVE_CONTROLLER_OPERATIONAL_SPEC_20261021.md).
Its recursive loop is:

```text
Phi_0(R) = Reason(R)
Phi_{k+1}(R) = Reason(
    R,
    Critique(Phi_k(R)),
    Counterexample(Phi_k(R)),
    EvidenceSearch(Phi_k(R))
)
R* = Fix_R { argmax_k J(Phi_k(R)) }
```

The registered objective is:

```text
J(R) = E[U + gamma V]
     + lambda IG
     + rho C_consistency
     + sigma Q_self
     + eta DeltaCompetence_future
     - beta H
     - kappa Cost
```

The associated persistent update is:

```text
M_{t+1} = Compress(M_t union Evidence_t union R*_t)
```

The proposed training loss is separate from runtime recursion:

```text
L = L_answer + lambda_c L_correction + lambda_v L_verification + lambda_s L_stopping
```

## Implementation status

The exact formula is not implemented as a neural architecture. The repository
contains a bounded CPU controller in `tools/recursive_controller.py` and the
C0/C1 runner in `tools/run_recursive_controller_c0_c1.py`. Those implement
typed state, immutable transitions, finite budgets, synthetic critique,
counterexample/evidence actions, verification and quarantined memory writes.

They do not implement a learned `V`, calibrated information gain, learned
self-quality, future-competence feedback, neural fixed-point iteration or
trained stopping. The controller therefore operationalizes the contract but
does not demonstrate neural OP reasoning.

## Existing evidence

The C0/C1 results in
`experiments/crucible_recursive_controller_c0_c1_20261021/RESULTS_SUMMARY.md`
show:

* 11/11 CPU contract checks passed.
* Frozen synthetic replay preserved shortcut invariance.
* Verification-only reached 1.000 synthetic accuracy; adaptive reached 0.8008.
* Policy advantage was explicitly classified `not_established`.
* Future competence was disabled; no learning intervention occurred.

These are controller-contract results, not evidence that a neural Crucible
model has learned recursive reasoning.

## Provenance boundary

The equations are preserved in the repository as a conversation-origin
specification. The comparison audit records that the claimed original braid
exchange files were not found locally. The next pilot therefore uses the
repository specification exactly and labels the neural integration
exploratory; it does not claim recovery of an undocumented implementation.

## Pilot scope

The first implementation uses the actual Crucible dense latent-attention +
SwiGLU model and the existing CPU/GPU data path. It compares one-pass output
against an external, bounded OP controller before attempting to internalize
the controller in model weights. The five prepared reasoning datasets are
used with the local TinyStories context corpus as a clearly labelled language
foundation proxy. FineWeb-Edu, Dolma and Gutenberg are not silently treated as
present; their absence is recorded in the data manifest.
