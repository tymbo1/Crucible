# Verified-procedure curriculum pilot

This exploratory run tests whether procedures produced and independently
verified by Crucible's CPU candidate path can be handed into later GPU
training batches.

Command shape:

```text
python3 curriculum_learning_pilot.py \
  --context-root <context-128-root> \
  --output-dir results_curriculum_pilot_gpu \
  --model-size tiny --steps 48 --target-count 16 \
  --device cuda --workers 4 --window 8
```

Arms:

* `gpu_no_curriculum`: fixed Crucible byte-LM batches only.
* `cpu_work_no_curriculum`: identical verified CPU work, discarded before
  training; this controls for scheduling and CPU workload.
* `verified_curriculum`: independently verified AST records enter later GPU
  batches in deterministic order.

The exploratory target slice is drawn from the frozen representative grammar
pool after removing source/validation functional hashes and filtering for
context feasibility. It is not the sealed Track B target set.

The run is not a registered Track B experiment and does not establish
cumulative learning. The full measurements are in `CURRICULUM_PILOT.json` and
`CURRICULUM_PILOT.md`.
