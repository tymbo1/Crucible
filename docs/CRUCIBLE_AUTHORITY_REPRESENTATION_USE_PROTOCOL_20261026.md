# Crucible Authority Representation-versus-Use Diagnostic v1

Date: 2026-10-26  
Status: read-only checkpoint diagnostic

## Objective

Determine whether the verified A3 authority-aware role-binding checkpoints
represent the `EARLIER`/`LATER` authority cue, and whether that information is
used to select the operation pointer. Compare familiar authority layouts with
the position-safe unseen reversed-authority layout used by the accepted clean
counterfactual diagnostic.

## Frozen source

- Source experiment: `crucible_authority_aware_role_binding_20251025_v2`
- Checkpoints: the four A3 `AUTHORITY_POINTER_BALANCED` checkpoints only
- Familiar layouts: `AUTHORITY_NEAR`, `AUTHORITY_EARLY`, `AUTHORITY_MIDDLE`
- Unseen layout: `UNSEEN_REVERSED_AUTHORITY`
- No retraining, optimizer updates, architecture changes or sealed-test access

## Diagnostics

### 1. Authority-cue probes

Capture the embedding, each dense-block output and final-normalized hidden
state. Fit a ridge linear probe on the frozen authority-training records,
using only the `EARLIER`/`LATER` label. Evaluate separately on each familiar
validation layout and on the clean unseen counterfactual records. Probe
results are descriptive: decodability does not establish causal use.

### 2. Causal-access audit and activation patching

First compare hidden states and operation-pointer logits at the two operation
candidate positions. In the clean unseen layout, the authority field follows
both candidates, so a causal model should have no path from that later field to
those earlier positions. This access audit is required to interpret any
authority-pointer result.

For completeness, replace the hidden vector at the authority-token position
in one member with the corresponding vector from the opposite-authority member.
Perform this at the embedding, each transformer layer output and final-
normalization output. Record operation-pointer changes, pointer correctness
and logit changes. Because the authority token is later than the operation
pointer candidates in this layout, this is a bounded negative-control
intervention, not proof that authority is unused in every pathway.

Controls:

- Same-input self-patching at every patch site.
- Capture-only instrumentation with numerical equivalence to an ordinary
  forward pass.
- Hidden-state probes at both operation-candidate positions.
- No interpretation from a patch effect alone; patching can introduce other
  contextual differences.

### 3. Familiar-versus-unseen comparison

Report pointer accuracy and authority-probe accuracy separately for the three
familiar validation layouts and the position-safe unseen layout. Do not pool
the earlier fixed-width padded counterfactual export with this diagnostic.

## Required invariants

- Checkpoint SHA-256 values match the accepted v2 summary.
- Clean unseen counterfactual records have identical operand and operation
  positions within each pair and exactly reversed target operation.
- Source train and validation record IDs are disjoint.
- The A3 baseline pointer predictions exactly reproduce the accepted clean
  counterfactual export and the accepted familiar validation export.
- Capture-only hooks return unmodified tensors and reproduce logits within the
  recorded numerical tolerance.
- Same-input self-patching produces no material output change within tolerance.

## Interpretation boundary

An authority probe can show that a cue is linearly recoverable, not that the
model uses it. A failed probe cannot prove that the information is absent. An
activation-patching effect can show causal sensitivity under the intervention,
but not by itself identify a unique internal mechanism. Any familiar/unseen
difference remains conditional on the tested layouts and training distribution.
