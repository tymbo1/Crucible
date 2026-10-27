# Scientific Interpretation

This experiment holds the M2 candidate panel, context pool, serialization,
dense SHARED_SCORER, cyclic schedule, optimizer path, 500-update budget and
row count fixed while increasing nested training-pair coverage from D4 to D8
to D12. D4 is an immutable M0->M2 crossover reference; D8 and D12 are new
training arms.

The primary behavioral question is whether broader pair coverage improves
transfer after training competence is established. The graph descriptors are
included to make structural changes auditable, but this task directly asks for
identifier matching and does not require transitive paths. Since the density
conditions also change degree, connectivity and finite distances, this run
cannot identify graph topology or path-length causality.

See SUMMARY.json, ERROR_BREAKDOWN.json and each cell's validation trace for
complete denominators and seed variation. No architecture, tokenizer,
optimizer, schedule or sealed-test boundary was changed.
