# Crucible 170M persistent-memory factorial — 100-update smoke run

Execution completed locally on the RTX 3060 Ti in 3,732.5 seconds.

- Model: unchanged 170,233,600-parameter E16 Crucible model.
- Seeds: 42 and 1337, using the existing frozen anchors.
- Arms: baseline, memory-only, skip-only and full.
- Five continuation phases of 20 attempted updates each.
- No cloud resources or checkpoint downloads.

Lower bits/byte is better.

| Arm | Mean final validation bits/byte | Mean raw-byte NLL | Applied updates | Skipped updates |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.94635 | 2.04226 | 20 + 20 | 0 + 0 |
| memory-only | 2.97418 | 2.06154 | 20 + 20 | 0 + 0 |
| skip-only | 2.95537 | 2.04850 | 20 + 20 | 0 + 0 |
| full | 2.96759 | 2.05697 | 20 + 20 | 0 + 0 |

Memory-only and full recorded 2,568 and 2,564 prior-phase teacher hits for
seeds 42 and 1337 respectively. The skip detector produced no skips because
the fixed maturity threshold is eight observations while this shortened run
provides only five phases.

The result is therefore an integration smoke test, not evidence that the
170M model benefits from persistent memory or redundant-update skipping. At
this budget, all intervention arms were slightly worse than baseline on mean
validation bits/byte, and the skipping component was not exercised. The full
50,000-update factorial remains intentionally unrun because its preflight
estimated approximately six days locally.
