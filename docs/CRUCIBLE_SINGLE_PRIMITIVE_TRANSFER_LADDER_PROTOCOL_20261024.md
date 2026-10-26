# Single-Primitive Transfer Ladder

Date: 2026-10-24  
Status: frozen before training  
Hardware: CPU only

## Objective

Localize the supplied-rule execution failure before introducing adaptation loops, symbolic/neural hybrids or neuromorphic mechanisms.

The ladder increases one demand at a time:

1. one canonical primitive with a fixed parameter;
2. diversity of input operands;
3. diversity of explicit rule parameters;
4. two-step composition;
5. three-step composition.

The model receives the rule explicitly. This is execution/transfer, not rule induction.

## Neural controls

- Existing dense Stage-A control, unchanged architecture and tokenizer.
- Seeds `42` and `1337`.
- 500 optimizer updates per tier and seed.
- Checkpoints/metrics at steps 0, 100, 250 and 500.
- Same optimizer, learning rate, initialization convention and raw-byte answer interface as the preceding cognitive experiments.
- No GPU, E4 routing, test-set evaluation or cloud resources.

## Tiers

| Tier | Supplied rule | Transfer being tested |
|---|---|---|
| `single_fixed` | `y=(x+1) mod 10` | Unseen input operands `x=8,9` after training on `x=0..7` |
| `single_operand` | `y=(x+z+1) mod 10` | Unseen `(x,z)` combinations while each digit value is represented in training |
| `single_parameter` | `y=(x+z+c) mod 10` | Unseen explicit parameter values `c=5..9` after training on `c=0..4` |
| `depth_2` | `y=op2(op1(x))` | Two explicit operations, with held-out input values but familiar operation parameters |
| `depth_3` | `y=op3(op2(op1(x)))` | Three explicit operations, with held-out input values but familiar parameters |

## Positive controls and safeguards

- A deterministic symbolic executor computes every target independently.
- Oracle audit must report zero target mismatches and zero train/validation prompt overlap.
- A tier is not interpreted as a neural failure unless its symbolic executor reaches 100%.
- Training and validation records are generated before training and hashed.
- Results are reported by tier, transfer group and seed.
- No original sealed test partition is opened.

## Measurements

For every checkpoint and final model:

- training exact accuracy;
- transfer validation exact accuracy;
- token accuracy and target loss;
- per-input/per-parameter group accuracy;
- prediction digit histograms;
- symbolic-oracle accuracy;
- elapsed CPU time and parameter count.

## Interpretation gates

- If the canonical fixed tier fails despite a 100% symbolic oracle, investigate interface/optimization before arithmetic or composition.
- If canonical transfer passes but operand diversity fails, investigate representation of multiple operands.
- If operand transfer passes but unseen parameters fail, investigate parameter binding.
- If single-primitive transfer passes but depth-2/3 fails, composition is a defensible bottleneck.
- No tier result authorizes GPU scaling or hardware integration by itself.
