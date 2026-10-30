{
  "base_checkpoint": {
    "path": "<step-15000-checkpoint>",
    "sha256": "e49faa226aaa96bc26f4abbed700ecf313ad59cbf68869cfc0327c61f5902d5a",
    "step": 15000,
    "training_tokens": 25270547
  },
  "classification": "bounded_source_competence_phase_only",
  "data": {
    "manifest_sha256": "d0534a7dcc083bd31b1424d298baacde9f278f4ef00345b3fcf3df5305bc6cd3",
    "sealed_answer_files_loaded": false,
    "source_train_sha256": "88ec12a6884e30278807f5ca677fe7dba7dd92a17675d9f7a24c6ca4f72efa65",
    "source_validation_sha256": "b601dfb2afa13ad5faa5acf26b8dbbca7ba9c11c36d0f8ff719e6078568e58d3",
    "target_data_loaded": false
  },
  "limitations": [
    "This phase tests source competence only; it does not test C0--C3 productive transfer.",
    "The Boolean prompt encoding and source fine-tuning settings are recorded for reproducibility and must remain fixed for interpretation."
  ],
  "model": {
    "parameter_count": 120842496,
    "state_digest": "ab316e40d942e7e5560c0c6f21ff293251bb3f1fb0d0121d15bf7d090dc52f0a"
  },
  "safety": {
    "checkpoint_integrity_checked_before_save": true,
    "finite_model_and_optimizer_at_each_update": true,
    "hard_stop_after_update_1000": true
  },
  "schema_version": "crucible-productive-retention-source-gate-v1-20261031",
  "scientific_authorization": false,
  "sealed_target_evaluation_started": false,
  "seed": 42,
  "source_gate": {
    "final": {
      "accuracy": 0.375,
      "correct": 6,
      "gate_pass": false,
      "rows": [
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0000"
        },
        {
          "correct": true,
          "expected": "0",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0001"
        },
        {
          "correct": true,
          "expected": "0",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0002"
        },
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0003"
        },
        {
          "correct": false,
          "expected": "1",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0004"
        },
        {
          "correct": true,
          "expected": "0",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0005"
        },
        {
          "correct": false,
          "expected": "1",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0006"
        },
        {
          "correct": true,
          "expected": "0",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0007"
        },
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0008"
        },
        {
          "correct": false,
          "expected": "1",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0009"
        },
        {
          "correct": false,
          "expected": "1",
          "prediction": "0",
          "raw_generation": "0\n",
          "task_id": "source|seed42|source_validation|0010"
        },
        {
          "correct": true,
          "expected": "1",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0011"
        },
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0012"
        },
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0013"
        },
        {
          "correct": false,
          "expected": "0",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0014"
        },
        {
          "correct": true,
          "expected": "1",
          "prediction": "1",
          "raw_generation": "1\n",
          "task_id": "source|seed42|source_validation|0015"
        }
      ],
      "trials": 16
    },
    "minimum_correct": 13,
    "pass": false,
    "threshold": 0.8,
    "trajectory": [
      {
        "evaluation": {
          "accuracy": 0.625,
          "correct": 10,
          "gate_pass": false,
          "rows": [
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0000"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0001"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0002"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0003"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0004"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0005"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0006"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0007"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0008"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0009"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0010"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0011"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0012"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0013"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0014"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0015"
            }
          ],
          "trials": 16
        },
        "update": 250
      },
      {
        "evaluation": {
          "accuracy": 0.4375,
          "correct": 7,
          "gate_pass": false,
          "rows": [
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0000"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0001"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0002"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0003"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0004"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0005"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0006"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0007"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0008"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0009"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0010"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0011"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0012"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0013"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0014"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0015"
            }
          ],
          "trials": 16
        },
        "update": 500
      },
      {
        "evaluation": {
          "accuracy": 0.5,
          "correct": 8,
          "gate_pass": false,
          "rows": [
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0000"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0001"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0002"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0003"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0004"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0005"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0006"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0007"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0008"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0009"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0010"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0011"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0012"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0013"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0014"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0015"
            }
          ],
          "trials": 16
        },
        "update": 750
      },
      {
        "evaluation": {
          "accuracy": 0.375,
          "correct": 6,
          "gate_pass": false,
          "rows": [
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0000"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0001"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0002"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0003"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0004"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0005"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0006"
            },
            {
              "correct": true,
              "expected": "0",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0007"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0008"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0009"
            },
            {
              "correct": false,
              "expected": "1",
              "prediction": "0",
              "raw_generation": "0\n",
              "task_id": "source|seed42|source_validation|0010"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0011"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0012"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0013"
            },
            {
              "correct": false,
              "expected": "0",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0014"
            },
            {
              "correct": true,
              "expected": "1",
              "prediction": "1",
              "raw_generation": "1\n",
              "task_id": "source|seed42|source_validation|0015"
            }
          ],
          "trials": 16
        },
        "update": 1000
      }
    ],
    "trials": 16
  },
  "source_gate_authorized": true,
  "source_training": {
    "batch_size": 8,
    "context": 512,
    "elapsed_seconds": 333.1004709940171,
    "examples": 8000,
    "examples_per_second": 24.01677780919046,
    "learning_rate": 0.0003,
    "peak_vram_allocated": 4679168000,
    "peak_vram_reserved": 5039456256,
    "precision": "bf16",
    "updates": 1000
  },
  "target_training_started": false
}
