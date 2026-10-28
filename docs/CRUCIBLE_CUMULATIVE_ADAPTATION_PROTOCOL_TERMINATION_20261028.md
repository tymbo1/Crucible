# Cumulative adaptation protocol termination record

Date: 2026-10-28
Status: **TERMINATED BY USER before registered scientific execution**

## Decision

The registered cumulative-adaptation protocol was terminated before its
scientific execution gate was used. The original protocol and amendment are
retained unchanged as historical records. They must not be represented as a
preregistration for the exploratory run below.

The subsequent run was explicitly authorised as a post-protocol exploratory
execution using reconstructed deterministic manifests and newly written
exploratory target bundles.

## Exploratory execution

- Artifact directory: `experiments/crucible_cumulative_adaptation_exploratory_20261028_v3`
- Classification: `exploratory_post_protocol`
- Target task-seed units: 288
- Arms: A0, A1, A2, A3
- Target-arm rows: 1,152
- GPU allocation: none
- Sealed answers: opened once, only after the proposal digest was fixed
- Registered success criteria: not applied
- Registered statistical inference: not applied
- CPU path: SSRE ordered process reservoir, four workers, bounded window eight
- CPU accelerator reference SHA256: `105c3120d07a4d29e9371713de12829aa913a68e557f0507f3ad93336a25f0f1`

The exploratory target query pairs were generated deterministically after
termination. This is an explicit protocol difference and limits the result to
descriptive evidence.

## Descriptive outcome

No operator reached the fixed 16/16 promotion requirement. A3 therefore
remained `unactivated_inconclusive`, as required by the frozen zero-promotion
rule. The four arms had identical target outcomes in this run: 120 accepted
proposals and 73 post-reset lifecycle successes out of 288 target tasks.

This does not establish cumulative policy adaptation, and it does not provide
evidence against an active adaptive policy because the adaptive mechanism was
not activated. It is not a valid preregistered efficacy result.

## Reproducibility identifiers

- Input bundle SHA256: `31a43e1f7da4d9b2310c8cd81d635e770adc0f531bca394004c71d721a01a75c`
- Answer bundle SHA256: `ce107c656542bf3b79afe21778292d0dda65f0a3dc654b481a04fab8560cf485`
- Proposal digest: `ffb51dda8f81935621aa1c9e35553dd2758c012f055ce767ebd63b14179a9082`
