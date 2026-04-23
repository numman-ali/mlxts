# Runtime Review: Long-generation chat streaming

## Summary

This change hardens the text-stream helper and cache-view path used by
`examples/chat` for long responses. The previous implementation in
`packages/transformers/src/generation.ts` re-decoded the full generated
continuation on every token, then diffed the new string against the prior full
string to emit a delta. That kept correctness, but it turned long outputs into
repeated O(n) decode work and growing host-side string churn.

The fix keeps the exact final text contract while batching stream flushes. We now
re-decode the full continuation every 32 generated tokens plus once at the end,
instead of once per token. That preserves correctness, reduces long-output host
pressure substantially, and keeps the public surface unchanged.

The deeper cache fix aligns our async decode path with the reference repos. MLX-LM
does use one-token async lookahead, but each cache update returns a fresh prefix
slice handle to attention. Our cache path was reusing and retargeting prefix view
handles with `sliceViewInPlace`, which made the aliasing contract sharper than the
reference path and harder to reason about under async lookahead. The cache
runtime now returns stable per-step prefix views and keeps async lookahead
enabled.

## Files Reviewed

- `packages/transformers/src/generation.ts`
- `packages/transformers/src/generation.test.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/view.test.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`

## Tensor Lifetime Audit

The streaming patch stays in the host-side text layer and does not add new
native tensor ownership. The cache patch intentionally changes ownership of
prefix cache views: when the visible prefix is shorter than the backing cache
capacity, the cache runtime now returns a fresh owned slice view to the attention
call, and the `using` cache-view disposal releases it after the forward pass.
When the visible prefix is the whole backing buffer, the cache still returns a
borrowed full-buffer view owned by the cache state.

I re-read the surrounding generation/runtime code to confirm the change does not
alter prompt embedding ownership or the existing `generateTokensInternal()` final
cleanup boundaries. The focused cache-view regression keeps a borrowed prefix
view alive across a later cache mutation and proves it remains stable.

## Memory / Performance Evidence

Validated with:

- `bun test packages/transformers/src/generation.test.ts packages/transformers/src/load.test.ts`
- `bun test packages/transformers/src/infrastructure/cache/view.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/generation.test.ts packages/transformers/src/load.test.ts`
- `bun run typecheck`

Added coverage:

- `packages/transformers/src/generation.test.ts` now verifies that
  `generateTextStream()` batches decode work for longer continuations instead of
  calling `decode()` once per generated token.
- `packages/transformers/src/infrastructure/cache/view.test.ts` now verifies
  that a prefix cache view handed to attention remains stable across a later
  cache mutation.

Manual reproduction evidence before the fix:

- Core cached generation with `generateTokensInternal(..., { maxTokens: 32000, eosTokenIds: [] })`
  did not immediately fail, which weakened the “large `maxTokens` alone breaks
  the cache loop” hypothesis.
- The text-stream path showed the stronger regression shape: it visibly slowed
  as continuation length grew because it re-decoded the full prefix on every
  token.

Reference comparison:

- `.reference/mlx-lm/mlx_lm/generate.py` uses async one-token lookahead, but
  `.reference/mlx-lm/mlx_lm/models/cache.py` returns fresh cache slices from
  `update_and_fetch()` rather than retargeting a previously returned view handle.
- `.reference/transformers/src/transformers/generation/streamers.py` streams
  text from a token cache and flushes finalized pieces instead of re-decoding
  the entire response as the primary public streaming contract.
- `.reference/transformers/src/transformers/cache_utils.py`,
  `.reference/omlx/omlx/memory_monitor.py`, and
  `.reference/vllm-mlx/docs/reference/configuration.md` all reinforce that
  serious long-generation stacks need explicit cache policy: static/dynamic or
  quantized cache, memory estimates, paged cache, or cache memory budgets.

This review does not claim that every possible 32k-token chat run is now solved.
Long outputs still grow prompt-cache state, and extremely large decode runs can
still hit real KV-memory pressure.

## Independent Review

Independent sub-agent audits informed this fix:

- `Aristotle` audited the chat and streaming surface and identified the
  full-prefix re-decode path plus the external prompt-cache retention pattern as
  the highest-confidence long-output risks.
- `Turing` independently audited the cache/runtime path and agreed that the
  text-stream layer was a concrete long-output weakness, while also flagging a
  deeper second-tier risk around async lookahead with borrowed cache views.
- `Galileo` checked that second-tier cache hypothesis against the runtime shape
  and reference repos. It did not prove async lookahead itself was unsafe, and
  recommended keeping lookahead enabled while making cache views stable per step.

Those reviews were independent of the implementation pass and helped separate
the confirmed stream bug from the deeper cache-view sharp edge.

## Remaining Risks / Follow-ups

- Very large `--max-tokens` values still imply large live cache state. For
  Qwen3.6 specifically, long decode runs remain capable of hitting legitimate
  memory limits even after the stream fix.
- `examples/chat` still uses a long-lived external prompt cache with no turn
  budget or warning. That is semantically correct, but it is not yet a
  user-friendly memory policy for extreme runs.
- The next ergonomic/runtime tranche should add cache-budget controls similar in
  spirit to MLX-LM `max_kv_size` / KV quantization and serving stacks' paged or
  memory-aware cache policies.
