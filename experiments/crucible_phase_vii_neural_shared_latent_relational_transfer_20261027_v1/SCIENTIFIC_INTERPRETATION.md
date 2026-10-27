# Scientific interpretation

Phase VII is a valid negative transfer result under the tested conditions.
All eight N1/N2 arms passed the 95% row-level training-competence gate, so the
held-out comparison is interpretable as transfer rather than an acquisition
failure. Nevertheless, every N1 and N2 arm was below the matched frozen N0
reference on paired validation correctness. Pooled paired correctness was
86/192 for N0, 59/192 for N1 and 53/192 for N2.

The preregistered hypothesis that the shared-latent neural representation
would improve transfer was therefore not supported in this implementation.
The additional cycle-consistency objective also did not meet its preregistered
benefit criterion. The result does not show that the broader architectural
idea is impossible: N1/N2 use a structured parsed representation and a
different computational pathway from N0, and only two graph families, two
seeds and known-identifier held-out pairs were tested.

The result establishes a sharper boundary: supplying a shared relational
inductive bias and training it to competence was insufficient to produce
better transfer on this benchmark. The next investigation should therefore
audit the learned representation and query readout, rather than claim that
symbolic Phase VI success has transferred to neural Crucible.

The executed matrix was eight trained arms (two variants × two graph families
× two seeds). The protocol's expected-cost paragraph described four new arms;
that documentation inconsistency is recorded in the final report and does not
alter the executed conditions.
