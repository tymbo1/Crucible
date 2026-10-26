# Scientific Interpretation

This is a matched fixed-slot versus shared-scoring diagnostic. Both learned
variants use the same dense backbone family and train on two- and three-
candidate examples. Transfer evaluates four, five and eight candidates. The
deterministic identifier oracle checks that the generated task and evaluator
are solvable.

The shared scorer is the candidate-independent hypothesis. Training competence
must be established before learned transfer is interpreted. None of the eight
learned arms passed the 95% training gate, whereas the deterministic oracle
passed 2,064/2,064 rows. The task and evaluator are therefore valid, but this
run does not distinguish learned candidate extraction from learned matching or
show a shared-scorer advantage. Paired counterfactual correctness remains the
primary metric for a future competent run.
