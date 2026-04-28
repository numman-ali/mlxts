# @mlxts/protocols

Zero dependencies. `@mlxts/core`, `@mlxts/nn`, `@mlxts/transformers`, fetch, streams, FFI, and model state are off-limits.

The current surface is reasoning-tag normalization: `splitReasoningTags`, `createReasoningTagStream`, `cleanReasoningFromText`, and the `ReasoningText` and `ReasoningTextDelta` types.

This package keeps wire-format normalization out of any package that owns side effects. `@mlxts/serve` and `@mlxts/agent` consume from here. Duplicating reasoning-tag logic in either is forbidden.

A helper is promoted from `@mlxts/serve` into here only when both `@mlxts/serve` and `@mlxts/agent` need it. Pre-emptive widening is forbidden.

`REASONING_TAGS` is the canonical tag list. New model families that emit known reasoning tags add entries here.
