# Crucible GitHub Artifact Policy

The GitHub repository stores the reproducibility surface: protocols, source
tools, reports, summaries, predictions, manifests, audits, hashes and complete
metadata for the latest accepted experiments.

Model checkpoints and raw/intermediate traces remain outside the normal Git
history because the workspace contains approximately 109 GiB of checkpoints
and several raw files exceed GitHub's 100 MiB file limit. Their paths, sizes and
SHA-256 digests are recorded in
[CRUCIBLE_GITHUB_ARTIFACT_MANIFEST_20261026.json](CRUCIBLE_GITHUB_ARTIFACT_MANIFEST_20261026.json).

Regenerate the manifest with
`tools/build_crucible_github_artifact_manifest_20261026.py` before archiving a
new local artifact set. A future large-artifact archive can be verified against
the same manifest without bloating the source repository.
