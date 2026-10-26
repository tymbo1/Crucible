# Crucible C2 readiness audit

Date: 2026-10-22

## Decision

C1r is complete and remains frozen. C2 is not executable from the current
repository without adding a valid frozen-model task adapter and a cognitive
task checkpoint. No C2 model evaluation was launched.

Running the TinyStories-trained 170M anchor directly against the cognitive
manifests would not be a valid C2 result: it would primarily measure missing
task competence and prompt adaptation, not the value of recursive critique or
verification.

## C1r status

The sealed C1r prediction/intervention experiment is intact:

- 7 indexed artifacts; all SHA-256 checks pass.
- Prediction testing reduced harmful transfer relative to corroboration-only
  and resolved its truthful prediction-only condition.
- Intervention testing reduced harmful transfer relative to
  corroboration-only and resolved its truthful intervention-only condition.
- Inconclusive and unavailable tests caused abstention.
- The experiment is explicitly hand-coded symbolic testing, not evidence that
  a neural model learned causal reasoning.

## C2 requirements versus available assets

| Requirement | Current status |
|---|---|
| Corrected v2 cognitive manifests and exact oracles | Available and validated |
| Frozen model checkpoint evaluated without weight updates | No cognitive-task checkpoint in this repository |
| Task-to-model input/output adapter | Not implemented |
| Competent frozen model reference | Not established |
| One-pass, fixed-depth, verification-only and adaptive policies | Symbolic controller policies exist; neural integration does not |
| Matched answer/tool/compute accounting | Defined in the C2 specification; not yet instrumented for a model |

The existing 170M anchors are TinyStories language-model states. The recent
500-update scratchpad branch was an integration test and deliberately did not
save a cognitive-task checkpoint; its results remain a separate frozen branch.

## Next valid gate

The smallest defensible next action is the previously specified Stage A
instrumentation pilot: a disposable 20-step capture preflight followed by a
bounded E4-versus-dense learnability pilot, with state save/reload and expert
intervention checks. It must establish a task-competence floor and a working
adapter before C2 compares recursive policies.

Until that adapter and competence floor exist, C2 remains a preparation gate,
not a neural reasoning result. No cloud execution, checkpoint download or
architecture change is required for this audit.
