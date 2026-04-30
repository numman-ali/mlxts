# Qwen Decoded Image Cache

## Summary

`@mlxts/serve` now keeps a host-side LRU cache for decoded and resized RGB
image bytes used by Qwen image serving. The cache stores only cloned
`Uint8Array` RGB data plus dimensions, never MLX tensors, prepared prompts, or
vision embeddings. Qwen image cache keys are derived from the image byte
SHA-256 digest and Qwen preprocessor config, not from mutable URL strings.

## Files Reviewed

- `packages/serve/src/media/decoded-image-cache.ts`
- `packages/serve/src/engine/content.ts`
- `packages/serve/src/media/decoded-image-cache.test.ts`
- `packages/serve/src/engine/content.test.ts`

## Runtime Sensitivity

The changed production path runs during host-side media loading before the
model lane. It can skip repeated platform decode and resize work for identical
image bytes under the same Qwen preprocessor config. It does not change
generation routing, scheduler admission, cache tensor semantics, prompt-prefix
snapshot/fork behavior, or SSE streaming.

Remote image URLs still fetch bytes before cache lookup because URL content is
mutable. The cache is content-addressed after the allowed transport policy has
accepted and bounded the payload.

## Tensor Lifetime Audit

No tensor-producing calls changed. The decoded image cache stores only host
`Uint8Array` RGB bytes and returns clones on read. Qwen tensor creation remains
inside `prepareQwen3_5ImageBatch()` and is still disposed in the existing
`preparePrompt()` `finally` block.

## Memory / Performance Evidence

The cache is byte-budgeted with LRU eviction. Entries larger than the whole
budget are skipped, so unusually large decoded images do not become sticky
resident host memory. Cache hits avoid repeated `sips` resize/decode work for
the same image digest and preprocessor config while preserving the existing
media-aware prompt-prefix cache identity.

## Tests

- `bun test packages/serve/src/media/decoded-image-cache.test.ts packages/serve/src/engine/content.test.ts packages/serve/src/media/image.test.ts`
- `bun test packages/serve/src/media/decoded-image-cache.test.ts packages/serve/src/engine/content.test.ts packages/serve/src/media/image.test.ts packages/serve/src/media/remote-image.test.ts`
- `bun test packages/serve`
- `bun run typecheck`
- `bun run lint`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run validate`
- `bun run regression:qwen-image -- --report-dir .tmp/qwen-decoded-image-cache`

## Independent Review

Kant reviewed the host-side cache boundary and called out the need to hash
bytes before `sips` size/decode work, keep the cache per engine/model-load,
return cloned RGB bytes, refetch mutable remote URLs, and avoid cache
population on failures or aborts. The implementation follows that shape and
adds tests for byte-budget LRU behavior, mutation isolation, zero-budget
disablement, preprocessor-key misses, remote refetch with same-byte decoded
cache hits, and rejected payloads.

## Out-of-scope Drift Noticed

Remote URL transport reuse remains out of scope without validator-aware content
freshness semantics. Persistent visual embedding caches, MLX tensor caches, and
multimodal batching remain separate future tranches.

## Remaining Risks / Follow-ups

The cache avoids host decode/resize work, not Qwen vision forward work. Exact
repeated prompts can already skip visual prompt tensors only when the
media-aware prompt-prefix cache covers the image-token prefix.
