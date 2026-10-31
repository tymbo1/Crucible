# CNSS Skill D lambda acquisition analysis v1

This report reanalyzes the existing rank-32 rows from the authorized D
capacity-versus-horizon control. It adds no training and changes no checkpoint.

The existing experiment already tested λ = 0.1, 0.4, 0.9 and 1.0 across three
seeds at both 2,000 and 8,000 updates. Therefore a duplicate lambda training
run would not add a new intervention.

The analysis compares training accuracy, held-out accuracy, canonical-state
displacement, cosine alignment and decoder margin. The D ≥95% gate and partner
gate remain closed.
