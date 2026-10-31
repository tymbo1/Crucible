# CNSS partner-generalization gate audit

Classification: **INCONCLUSIVE_PREREQUISITE_FAILURE**

The run completed with finite tensors and valid checkpoint dependencies, but the preregistered individual-skill gate was not met.

| Skill | Operation | Maximum exact test accuracy across lambda and seed |
| --- | --- | ---: |
| A | rotate-left by 1 | 1.0000 |
| B | XOR with 0x59 | 1.0000 |
| C | high/low nibble swap | 1.0000 |
| D | modular addition by 0x37 | 0.6875 |

The required individual-test threshold was 0.95. Because skill D did not qualify, the held-out partnership results are descriptive only. They cannot yet distinguish sender-specific, receiver-specific or pair-specific compatibility.

Engineering integrity checks passed:

- 240/240 raw checkpoints were present.
- Every checkpoint referenced the exact frozen workspace digest.
- No checkpoint header had a mismatched workspace digest or invalid step.
- Fresh adapter reload reproduced the recorded individual outputs.

The appropriate next step is a separately registered diagnosis of why modular addition was not acquired under this adapter, workspace and 2,000-update budget. No adaptive contract, transition bridge or partnership-specific conclusion is authorized from this result.
