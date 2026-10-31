# Crucible frozen source-competence diagnostic

Status: **completed — frozen inference only**

Checkpoint step: 15000
Device: cuda
Precision: bf16_autocast
Elapsed seconds: 4.31

| Seed/split | Category | Primary exact | Primary coverage | Forced-binary exact |
|---|---|---:|---:|---:|
| seed_1337/execution_test | explicit_execution | 0/16 | 0.000 | 9/16 |
| seed_1337/execution_validation | explicit_execution | 0/16 | 0.000 | 10/16 |
| seed_1337/source_function_validation | function_identifiable_induction | 0/16 | 0.000 | 4/16 |
| seed_1337/source_test | query_identifiable_induction | 0/32 | 0.000 | 14/32 |
| seed_1337/source_validation | query_identifiable_induction | 0/32 | 0.000 | 14/32 |
| seed_42/execution_test | explicit_execution | 0/16 | 0.000 | 11/16 |
| seed_42/execution_validation | explicit_execution | 0/16 | 0.000 | 7/16 |
| seed_42/source_function_validation | function_identifiable_induction | 0/16 | 0.000 | 7/16 |
| seed_42/source_test | query_identifiable_induction | 0/32 | 0.000 | 17/32 |
| seed_42/source_validation | query_identifiable_induction | 0/32 | 0.000 | 14/32 |

The full per-task results are in `FROZEN_DIAGNOSTIC_REPORT.json`.
No source training, target evaluation, productive-retention claim or architectural comparison was performed.

## Positive decoder control

{
  "correct": 16,
  "passed": true,
  "rows": [
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    },
    {
      "correct": true,
      "expected": "0",
      "prediction": "0"
    },
    {
      "correct": true,
      "expected": "1",
      "prediction": "1"
    }
  ],
  "total": 16
}
