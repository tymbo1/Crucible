# CNSS matched workspace-accessibility causal control v1

This protocol is registered but training is not authorized.

It tests one narrow causal comparison: the recovered nonlinear workspace
against a deterministic, shared signed-bit coordinate workspace with the same
256-dimensional interface. The alternative representation is operation-
agnostic: it encodes only the eight input bits and does not use the modular
addition rule, held-out targets or sealed test answers.

Both arms would use the existing rank-32 Skill D adapter, lambda 0.1, the same
paired step-zero adapter and optimizer states, the same candidate support,
standard batch schedule, seeds, 2,000 updates and fixed evaluation partitions.

The alternative workspace is deliberately a mechanism control, not a claim of
architectural equivalence. Its fixed encoder/decoder has no trainable
parameters, so wall-clock and workspace-compute costs must be reported. A
positive result would support workspace accessibility under this construction;
it would not establish general composability or lifelong learning.

No training, GPU execution or partner rerun is authorized by this protocol.
