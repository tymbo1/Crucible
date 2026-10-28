# Crucible reasoning-dataset preparation

Prepared 2026-10-29 for the future 500M Crucible curriculum. The raw bundle is
stored outside the Git repository at a local path recorded only in the private
preparation copy. External artifacts use the placeholder
`<local-dataset-bundle>`.

The repository contains provenance and hash metadata, not the raw datasets.
See [the machine-readable manifest](CRUCIBLE_REASONING_DATASETS_MANIFEST_20261029.json).

## Acquired resources

| Resource | Local material | Status | License record |
|---|---|---|---|
| MATH | Official source repository plus the data mirror referenced by that repository | Downloaded | MIT declared; verify source-problem rights before commercial training |
| BIG-Bench Hard | Official 23-task repository | Downloaded | MIT |
| SciBench | Official source repository with original JSON and solution files | Downloaded | MIT |
| ETHICS | Official source repository plus linked `ethics.tar`, extracted | Downloaded | MIT |
| BIG-Bench | Official repository, published 24-task BIG-Bench Lite subset | Downloaded | Apache-2.0 |

The BIG-Bench entry is intentionally the 24-task Lite subset, not the full
200-plus-task repository. This keeps preparation bounded while retaining the
published broad-reasoning control set.

## Storage and training boundary

The raw bundle is not committed or pushed to GitHub. The manifest records source
URLs, revisions, local paths, hashes and limitations so it can be reconstructed
without placing potentially restricted or bulky data into the code repository.

These datasets are currently preparation inputs, not an approved training run.
Before model training, create a derived-data manifest with exact filtering,
deduplication, split, answer masking and licensing decisions. In particular,
benchmark test answers must not silently enter the training corpus.

## Planned curriculum use

The saved curriculum plan remains future work: formal mathematics and logic,
then science, language/conceptual reasoning and finally normative evaluation.
The datasets will need conversion into Crucible learning episodes rather than
being concatenated blindly. Each episode should retain provenance, task family,
source split, answer visibility and verification status.
