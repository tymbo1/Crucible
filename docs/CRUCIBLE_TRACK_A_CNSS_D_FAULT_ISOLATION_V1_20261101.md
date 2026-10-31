# CNSS Skill D fault-isolation audit v1

This read-only extension examines the saved rank-32 modular-addition
checkpoints. It does not train, update, select or alter any checkpoint.

The audit separates three possible locations of failure:

1. the frozen encoder/decoder's ability to represent and reconstruct canonical
   eight-bit states;
2. the adapter's displacement from the canonical representation of the correct
   arithmetic result; and
3. the decoder's sensitivity to the adapter's actual intermediate state.

It uses Skill D's exact operation, `(x + 0x37) modulo 256`, and all 256 input
states. It reports carry-event and chain distributions, signed decoder margins,
canonical-to-actual interpolation, decoder-Jacobian projections, matched
controls for masks `0x80`, `0x60`, `0x40` and `0xC0`, and trajectories across
all saved rank-32 checkpoints.

The public output is aggregate-only. Per-input traces and displacement vectors
remain in the durable sealed artifact. The D competence gate and partner gate
remain unchanged.
