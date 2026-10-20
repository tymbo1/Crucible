# GPU Neural SSRE v0 measurement protocol

## Frozen inputs

The probe defaults to:

- `context_128` frozen Crucible package;
- seed 42;
- Branch B;
- E16, top-2, four MoE layers;
- the first complete optimizer-step group from the frozen schedule;
- BF16 autocast and the immutable seed anchor.

The model is loaded from the anchor twice only in the logical sense: one
unprofiled warm-up step is executed, then the same disposable model performs
one profiled forward/backward step. No optimizer update is performed, so the
anchor weights remain unchanged. The state digest is checked before and after.

## Measurements

The probe records:

1. per-layer/per-expert calls, selected-token counts, input shapes and token
   counts per call;
2. nested `moe/layer_*`, `expert/layer_*/expert_*` and
   `attention/layer_*` profiler ranges;
3. top CUDA operators by self device time;
4. overlapping indicative categories for matrix arithmetic,
   routing/indexing, layout/copy and activation operations;
5. synchronized host and CUDA time for the profiled forward/backward step;
6. input/config/anchor/source hashes and model-state equivalence.

Category totals are intentionally labelled overlapping. They must not be
added together or interpreted as mutually exclusive costs. Nested range totals
are likewise not additive: expert ranges sit inside MoE ranges.

The profiler captures the complete schedule step in one cycle. It does not
call `prof.step()` between microbatches, avoiding cycle clearing and preserving
all ranges for the report.

## Interpretation boundary

This gate does not claim a speedup, does not modify the frozen runner and does
not train a scheduler. It identifies whether the next implementation should
focus on grouped expert matrix operations, routing/dispatch, layout movement,
or another measured source of cost.

