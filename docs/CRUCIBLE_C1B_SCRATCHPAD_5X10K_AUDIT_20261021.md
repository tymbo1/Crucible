# C1b scratchpad continuation audit

This is a read-only post-run audit of
`experiments/crucible_recursive_controller_c1b_scratchpad_5x10k_20261021/`.
It does not modify the training artifacts.

## Scratchpad-disabled final evaluation

The final persistent-scratchpad model state for each seed was evaluated twice:
once with its scratchpad and once with scratchpad access disabled.

| Seed | Enabled accuracy | Disabled accuracy | Enabled mean cost | Disabled mean cost | Test scratchpad hits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.765625 | 0.765625 | 5.76953125 | 5.76953125 | 0 |
| 1337 | 0.76171875 | 0.76171875 | 5.50781250 | 5.50781250 | 0 |
| 2026 | 0.67578125 | 0.67578125 | 5.185546875 | 5.185546875 | 0 |

The shifted untouched test states did not match any mature scratchpad
association. Therefore the observed final-test difference from the paired
no-scratchpad arm cannot be attributed to direct test-time retrieval. The
scratchpad did affect training-time action selection and redundant-update
skipping, and those effects were carried into the final policy weights.

This does not establish that the scratchpad generalizes to new task families.
