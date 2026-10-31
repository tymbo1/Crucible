# CNSS D fault-isolation supplement

The canonical encoder/decoder reconstructs both every input state and every
correct modular-addition result state exactly: 256/256 in both cases.

The high-bit error groups have substantially larger adapter displacement than
matched correct controls:

| Error mask | Error displacement | Matched control | Error margin | Control margin |
|---:|---:|---:|---:|---:|
| `0x80` | 5.65 | 2.01 | −4.02 | 7.96 |
| `0x60` | 7.63 | 2.06 | −10.03 | 8.01 |
| `0x40` | 5.91 | 2.11 | −4.75 | 7.81 |
| `0xC0` | 7.85 | 1.83 | −7.15 | 8.77 |

Along the actual adapter-error direction, all four groups remain correct at
25% interpolation from the canonical state, then lose accuracy between 50% and
75%; at 100% interpolation they reproduce the observed failures. Displacement
norm and actual decoder margin have a mean final-checkpoint Pearson correlation
of approximately −0.94.

This is consistent with the adapter producing structured, noncanonical states
that eventually cross decoder boundaries. The mixed local boundary projections
mean the audit does not assign sole responsibility to the adapter or decoder.
It also does not establish that the network has learned a sequential carry
algorithm.
