# CNSS Partner Generalization and Contract-Locus Matrix v1

## Objective

This experiment tests whether neural composition requirements are primarily:

1. sender-specific;
2. receiver-specific; or
3. specific to the directed partnership between two skills.

It is a bounded follow-up to the four-point lambda sweep. It does not change or overwrite the original CNSS artifacts.

## Skills

The existing verified 8-bit workspace is reused. Four independently trained skills are used:

- `A`: rotate left by one bit;
- `B`: XOR with `0x59`;
- `C`: swap the high and low four-bit nibbles;
- `D`: add `0x37` modulo 256.

Each skill is trained independently at lambda values `0.1`, `0.4`, `0.9` and `1.0`, using three seeds and the registered 2,000-update budget. No composition examples are shown during training or validation.

## Partnership holdout

The six development partnerships are:

`A→B`, `B→C`, `C→D`, `D→A`, `A→C`, `B→D`.

The six held-out test partnerships are:

`B→A`, `C→B`, `D→C`, `A→D`, `C→A`, `D→B`.

Lambda or contract policies are selected from development partnerships only. Test partnerships are not used for selection.

## Leakage control

The primary pair metric uses sender test inputs whose transformed intermediate values are absent from the receiver's training and validation inputs. Full test-set composition accuracy is also reported, together with the collision count. This distinguishes partnership generalization from memorization of an intermediate numeric state.

## Selection policies

The report compares:

- one global lambda;
- sender-only lambda selection;
- receiver-only lambda selection;
- one shared lambda per skill;
- the complete sender-lambda/receiver-lambda compatibility tensor.

The pairwise tensor is descriptive. Selecting a separate lambda for every test pair is not treated as a deployable solution.

## Interpretation boundary

Evidence that a sender-selected policy transfers to unseen receivers supports a sender-level requirement. Evidence that most senders struggle with one receiver supports receiver sensitivity. Strong development performance but poor held-out partnership performance supports relationship-specific compatibility.

Positive results remain limited to this 8-bit mechanism probe. Independent seeds, larger state spaces, longer chains and additional operation families are required before making scalability claims.
