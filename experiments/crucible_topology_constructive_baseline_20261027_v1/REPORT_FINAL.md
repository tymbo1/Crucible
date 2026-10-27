# Crucible constructive relational baseline

Read-only symbolic replication of the Phase VI shared latent-potential proposal.

## Result

The supplied latent-potential representation and shortest-path composition recovered all eight held-out relations exactly for both committed graphs and all tested moduli (5, 11 and 97). The random lookup controls stayed near their expected 1/K accuracy. Injecting one ±1 edge corruption reduced shortest-path accuracy, with the diameter-5 graph lower than the prism for every tested modulus.

| Graph | K | Exact composition | Lookup | Corrupted shortest path | Corrupted consistency failures |
|---|---:|---:|---:|---:|---:|
| P12_PRISM | 5 | 8000/8000 (1.000) | 1605/8000 (0.201; expected 0.200) | 6797/8000 (0.850) | 1000/1000
| P12_PRISM | 11 | 8000/8000 (1.000) | 706/8000 (0.088; expected 0.091) | 6839/8000 (0.855) | 1000/1000
| P12_PRISM | 97 | 8000/8000 (1.000) | 92/8000 (0.011; expected 0.010) | 6883/8000 (0.860) | 1000/1000
| R12_DIAMETER5 | 5 | 8000/8000 (1.000) | 1647/8000 (0.206; expected 0.200) | 6471/8000 (0.809) | 948/1000
| R12_DIAMETER5 | 11 | 8000/8000 (1.000) | 730/8000 (0.091; expected 0.091) | 6592/8000 (0.824) | 954/1000
| R12_DIAMETER5 | 97 | 8000/8000 (1.000) | 78/8000 (0.010; expected 0.010) | 6602/8000 (0.825) | 950/1000

## Interpretation boundary

This establishes a constructive symbolic baseline, not neural discovery. The procedure was explicitly supplied, and all identifiers were present in the graph. It does not test new-identifier generalisation, learned latent representations, or whether Crucible implements path traversal, cycle checks or modular subtraction.

The corrupted-consistency check detects errors on cyclic edges. A bridge corruption in the diameter-5 graph can be absorbed by a component offset and is therefore not detected by cycle checks; this is an identifiability limitation of the symbolic graph, not evidence of successful correction.

See `RESULTS.json` for exact counts and consistency diagnostics.
