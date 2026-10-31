# CNSS D rank-32 generalization forensics v1

This is a read-only audit of the saved rank-32 D-skill checkpoints from the
capacity-versus-horizon control. It performs no optimizer updates, parameter
mutation, GPU execution or partner evaluation.

The audit evaluates all 256 eight-bit inputs against the independent oracle for
`(x + 0x37) modulo 256`, while preserving the original train/validation/test
partition. It reports exact accuracy, carry and wraparound strata, output-bit
accuracy, and one-input-bit transition consistency. Detailed per-input traces
are retained only in the durable sealed artifact; the repository report is
aggregate-only.

The result cannot promote D or unlock partner generalization. It is intended to
distinguish carry-sensitive, bit-local, partition-specific and shortcut-like
failure patterns before any further training is considered.
