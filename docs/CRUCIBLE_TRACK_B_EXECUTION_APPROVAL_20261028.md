# Crucible Track B execution approval

Date: 2026-10-28

The user explicitly authorized execution of the frozen B0--B3 scientific run
under amendment `6b2a8ad` with the instruction:

> Authorize the frozen B0–B3 scientific run under amendment 6b2a8ad.

The registered approval flags were set in:

- `docs/CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.json`
- `docs/CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_FREEZE_AMENDMENT_20261028.json`

The design, task manifest, seeds, fixed final checkpoint rule, two-GPU-hour
cap, leakage protections, lifecycle gates and sealed-test phase boundary were
not changed. The separate Verified Learning Operator Library experiment was
not approved or started.

Scientific execution begins only through the guarded runner after the
preflight and approval checks pass.
