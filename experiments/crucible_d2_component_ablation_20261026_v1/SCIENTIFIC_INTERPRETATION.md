# Scientific Interpretation

This is a matched 2x2 ablation. The accepted S4 cell supplies the successful
candidate-slot/decision-only control, while the accepted original D2 cell
supplies the operation-class/multi-task control. The two crossed cells isolate
the candidate-slot and objective factors as far as this small experiment can.

The primary evidence is paired counterfactual selection: changing only the
authority code must change the selected candidate. Training competence is
required before validation is interpreted.

All C1, C2 and C3 arms passed the training gate and achieved perfect reported
validation selection, counterfactual flips and two-way paired correctness.
C4 remained at approximately chance-level training selection and therefore is
not interpretable as a clean held-out generalization result. The operation
class readout was sufficient in C3, so the candidate-slot head is not required
for this two-candidate task. The auxiliary multi-task loss did not prevent
success in C2, but C4 shows that the original operation-class/full-multitask
configuration or its historical training trajectory remains a failure mode.

C4 is a reused historical control rather than a newly retrained crossed cell.
Consequently, this experiment does not identify whether the C4 failure is
caused by the auxiliary objective alone, its interaction with operation-class
readout, or the original training schedule/trajectory. A matched C4 replay
would be required for that narrower causal claim.

This experiment does not test larger candidate sets, arbitrary authority
relations or autonomous reasoning.
