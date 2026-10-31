# CNSS Skill D carry-balanced acquisition control v1

This protocol tests training-distribution frequency, not additional task
coverage. The balanced arm samples only from the existing 128-state D training
candidate support, preserving the fixed validation and 64-state test manifests
exactly.

Carry-event strata 0 through 8 receive deterministic quotas of
`[15,15,14,14,14,14,14,14,14]` per 128-draw balanced epoch. Sampling with
replacement is required because the existing strata contain unequal numbers of
candidate states; the candidate support remains identical to the existing
control.

The rank-32, lambda-0.1, 2,000-update control already exists for all three
seeds. The balanced arm must start from each corresponding persisted step-0
adapter checkpoint, so initialization is paired rather than regenerated.

This protocol is registered but not authorized for training. The D and partner
gates remain closed.
