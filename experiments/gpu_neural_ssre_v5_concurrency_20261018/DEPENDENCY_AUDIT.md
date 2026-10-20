# Crucible dependency audit for V5

Source: immutable `shared/model_core.py` from the context-gate package.

## Independent work

Within `ResearchMoELayer.forward`, after routing and token selection, each
expert/rank call is independent until its weighted result is accumulated into
the token output. This is the only candidate tested by V5.

Within one `SwiGLU`, `w1(x)` and `w2(x)` are independent, but `w3` waits for
both. This is a possible later kernel-fusion target, not part of this stream
preflight.

## Dependency-constrained work

- Transformer layers are sequential because layer `i+1` consumes layer `i`'s
  residual output.
- Within a layer, MoE waits for the attention residual and normalization.
- Attention output projection waits for attention output; attention waits for
  Q/K/V projections and masks.
- Routing must complete before expert token subsets are known.
- Expert output accumulation must complete before the layer residual is
  passed onward.
- Backward dependencies were not altered by this forward-only preflight.

## Integrity design

Stream fan-out computes each segment into a private tensor. The default stream
waits on every worker stream before performing ordered `index_put_` accumulation.
Therefore concurrent expert writes cannot race, and the accumulation order is
the same expert/rank order used by the V2 reference.

