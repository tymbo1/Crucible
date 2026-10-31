# CNSS D-skill corrective controls — post-run audit

The authorized CPU run completed 48 rows and produced all 264 expected durable
checkpoints. A separate fresh process verified finite payloads, adapter and
optimizer reloads, the exact frozen-workspace digest, and answer-free public
artifacts.

The D competence gate was not reached. The maximum held-out exact accuracy was
0.890625 (57/64), below the required 0.95 (61/64). Partner generalization
remains locked.

## Comparability limitation

The executed runner used condition-specific deterministic seed offsets:

`base_seed + 100000*skill_index + int(lambda*10000) + 910000 + rank*17 + updates`

The earlier partner runner used the same formula without the final `rank*17 +
updates` terms. Therefore the rank-8/2,000 rows are not exact initialization
replays of the earlier D run, and capacity-versus-horizon rows are not paired
from identical initial adapter draws.

The results remain valid as a fresh authorized acquisition control, but strict
paired causal claims about capacity versus training horizon are not made. A
seed-matched replication would need a separately registered protocol.

The original result and all checkpoint artifacts are preserved unchanged.
