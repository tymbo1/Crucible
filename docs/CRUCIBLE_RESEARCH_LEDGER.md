# Crucible Research Ledger

This ledger is the repository-backed scientific memory for Crucible. It records
what a specific tested hypothesis established, failed, or left unresolved. A
status applies only to the documented hypothesis under its stated conditions;
`FAILED` does not mean that the broader architectural idea is impossible.

Status values:

- `ESTABLISHED` — supported by the accepted artifact and its preregistered controls.
- `FAILED` — the stated hypothesis failed under the documented conditions.
- `INCONCLUSIVE` — informative result, but a confound prevents the narrower causal claim.
- `UNTESTED` — proposed, incomplete, or not yet supported by an accepted result.

The machine-readable source is [CRUCIBLE_RESEARCH_LEDGER.json](CRUCIBLE_RESEARCH_LEDGER.json).
Use `tools/search_crucible_research_ledger.py` before proposing an experiment;
search by mechanism, title, status or unresolved question. Validate the ledger
with `tools/validate_crucible_research_ledger.py`, and run
`tools/audit_crucible_research_ledger_coverage.py` to identify experiment
reports that still need ledger entries.

## Current high-value entries

| ID | Status | Finding |
|---|---|---|
| `authority-two-candidate-s4` | ESTABLISHED | A causally valid final decision interface can select between two candidates under S4. |
| `authority-component-ablation` | INCONCLUSIVE | C1/C2/C3 passed; historical C4 failed, but objective and trajectory are not isolated. |
| `shared-scorer-compositional-transfer` | FAILED | The tested dynamic scorer did not transfer reliably from three to four candidates. |
| `explicit-id-cardinality-transfer` | FAILED | The oracle passed, but none of eight learned arms passed training competence; learned cardinality transfer is therefore unresolved. |

The initial ledger seed covers the four highest-value experiments from this
research turn. The coverage audit currently finds 23 older experiment reports
awaiting structured extraction; they are deliberately not assigned statuses by
guesswork.
