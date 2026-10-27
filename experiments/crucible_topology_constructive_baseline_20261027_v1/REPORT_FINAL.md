# Crucible constructive relational baseline

Read-only symbolic replication of the Phase VI shared latent-potential proposal.

## Result

The supplied latent-potential representation and shortest-path composition recovered all eight held-out relations exactly for both committed graphs and all tested moduli (5, 11 and 97). The random lookup controls stayed near their expected 1/K accuracy. Injecting one ±1 edge corruption reduced shortest-path accuracy, with the diameter-5 graph lower than the prism for every tested modulus.

## Interpretation boundary

This establishes a constructive symbolic baseline, not neural discovery. The procedure was explicitly supplied, and all identifiers were present in the graph. It does not test new-identifier generalisation, learned latent representations, or whether Crucible implements path traversal, cycle checks or modular subtraction.

See `RESULTS.json` for exact counts and consistency diagnostics.
