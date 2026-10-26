# Crucible Dual-Track Research Programme

Date: 2026-10-25  
Status: adopted before execution  
Hardware: CPU only

## Purpose

Crucible research now proceeds on two independent tracks.

### Track A — Neural capability diagnostics

Preserve the existing 22M dense-control architecture, tokenizer, evaluator, seeds, checkpoints and sealed-test boundary. The immediate narrow control tests whether an explicit operation identifier can be recognized independently of executing the requested operand-selection procedure.

Track A results remain diagnostic. They do not establish self-learning or autonomous cognition.

### Track B — Learning Lifecycle v0

Build a minimal, separately testable procedural learner using a deliberately bounded symbolic hypothesis space. It must acquire an unfamiliar rule from demonstrations, verify it, retain a procedure and provenance, survive a temporary-context reset, execute on unseen inputs, and revise or invalidate knowledge after contradiction.

Track B is explicitly engineered scaffolding. Its success demonstrates a complete learning lifecycle, not neural meta-learning or general intelligence.

## Common interface

Both tracks use the following conceptual records:

- observation: input, output, source and provenance;
- hypothesis: candidate procedure, assumptions and compatibility evidence;
- learned procedure: versioned executable rule and applicability boundary;
- verification: independent checks and outcome;
- persistent state: durable procedure metadata, evidence references and revision history;
- evaluation: accuracy, coverage, uncertainty, revisions, interference and cost.

## Boundaries

- Preserve all previous experiments and checkpoints.
- No CMS2, ProjectBrain merge, fast weights, recursive controller or GPU in v0.
- No evaluator labels may be inserted into persistent state before the learner has made its decision.
- Raw-example retrieval is a negative control, not equivalent to procedural retention.
- Engineered symbolic inference must remain labelled separately from neural adaptation.

## Progression

The intended progression is:

`engineered learner → neural-assisted learner → adaptive learner → learner that improves its learning strategy`.

The first milestone is a reproducible lifecycle, not a claim that the neural control already possesses it.
