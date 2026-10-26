# Scientific Interpretation

This gate tests a dynamic candidate-pointer readout rather than extending the
fixed two-class S4 head. Training uses three candidates. Validation-3 changes
operand pairs and layout; Transfer-4 adds a fourth candidate that was absent
from training. Relation rules select a candidate by its operation marker, and
NONE requires abstention.

Training competence is required before transfer is interpreted. Counterfactual
flips and paired correctness are the principal relational metrics. Three of
four arms passed training competence, but Validation-3 transfer was weak and
Transfer-4 achieved 72/120 row-level pointer accuracy with 0/24 paired
counterfactual correctness. Abstention transferred 24/24, while positional
authority did not. The compositional transfer gate therefore failed despite
partial relation-matching behavior.

This is evidence of a bounded, structured failure—not evidence that the dense
backbone cannot represent candidate relations. The experiment changed the
readout deliberately; it did not change the dense backbone or tokenizer. It
does not establish arbitrary instruction hierarchy, general reasoning or
autonomous learning.
