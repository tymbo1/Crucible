# Scientific Interpretation

This diagnostic was designed to separate a representation burden from a
training-schedule contribution without reopening cardinality transfer. The
source split, labels, deterministic span metadata, one canonical interface,
one seed, optimizer family, update count and training gate were fixed before
execution. Structured cells use candidate/query/slot/count features but no
target or privileged equality label.

Only training competence is interpretable here. Validation and transfer were
intentionally not run. The full decision is recorded in REPORT_FINAL.md and
the machine-readable SUMMARY.json and ERROR_BREAKDOWN.json.

The dense cyclic cell reproduced the prior canonical shared-scorer result at
497/576. Deterministic shuffling reduced the dense result to 384/576, while
shuffling improved the smaller structured scorer from 390/576 to 432/576.
This supports a schedule contribution under the tested budget, but it is a
one-seed result and not a causal explanation. The structured cells were not
parameter-matched to the 22M dense model and neither passed the gate.

All four cells solved absent-query abstention at 192/192. The remaining
present-query matching task was the failure locus. Structured cells never
selected the 48 slot-1 targets, so the diagnostic does not support a claim that
explicit features made matching competent.

No cardinality transfer conclusion is permitted. The next controlled step is
to simplify and balance the two-candidate training target before reintroducing
mixed cardinalities.
