# Crucible source curriculum/evaluator repair

Status: **PASS** — CPU-only validation; no training or target evaluation.

Rows checked: 320
Distinct truth tables: 320
Distinct canonical structures: 320

## Controlled decoder

The repaired evaluator predicts exactly one `0`/`1` token from the fixed logit position immediately before the answer byte. Non-answer tokens and ties abstain; free-running generation is not used for the source gate.

```json
{
  "checks": {
    "fixed_answer_position": true,
    "non_answer_abstains": true,
    "nonfinite_rejected": true,
    "one": true,
    "ties_abstain": true,
    "zero": true
  },
  "passed": true
}
```

## Identifiability

Query-identifiable tasks require every grammar-consistent function to agree on the query. Function-identifiable tasks require exactly one grammar function to remain compatible. Ambiguous fixtures are detected and retained as an explicit negative test.

## Boundary

This artifact repairs measurement and curriculum construction. It does not establish neural competence, productive retention or target transfer. The existing C0–C3 experiment remains blocked.

## Errors

```text
none
```
