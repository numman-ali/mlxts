# Qwen Image Prefix Short Circuit

## Summary

Repeated Qwen image prompts can now check the media-aware prompt-prefix cache
before running request-local visual prompt preparation. When the retained cache
snapshot covers the image-expanded prompt prefix and the remaining suffix
contains no image tokens, serving skips Qwen image patchification, vision tower
forward, and prepared embedding scatter for that repeat. The generation step
still restores through the existing family-owned cache snapshot/fork path.

This is not a persistent visual-embedding cache. Serve does not retain
`MxArray` media tensors, and host media transport/decode remains request-local.

## Files Reviewed

- `packages/serve/src/engine/content.ts`
- `packages/serve/src/engine/content.test.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/engine.test.ts`
- `packages/serve/src/engine/prefix-cache.test.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional-support.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/preprocessing.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.test.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional-support.test.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/preprocessing.test.ts`
- `packages/transformers/src/index.ts`

## Runtime Sensitivity Notes

The runtime-sensitive surface is media prompt preparation before single-request
generation. The changed path does not alter decoder math, cache tensor layout,
sampling, scheduler admission, or protocol formatting.

The early skip is guarded by three conditions:

- the model-family adapter provides expanded token ids from transformer-owned
  Qwen grid/token helpers
- the existing prompt-prefix cache can reuse a retained snapshot for that
  expanded token sequence and media identity
- every token after the cached prefix is non-image text, so the suffix can be
  generated through token ids plus the restored cache

Qwen RoPE deltas remain family-cache-owned. A skipped repeat relies on the same
cached delta restoration path used after normal prepared-prompt cache hits.
The Qwen conditional test now compares explicit prepared suffix MRoPE position
ids against the token-only cached suffix path after a prompt-boundary snapshot.

## Tensor Lifetime Audit

Serve adds no long-lived tensor ownership. The early skip path creates no
prepared media tensors. The cold/fallback path still frees prepared image
`pixelValues` and `imageGridThw` in the existing `finally`, and generation still
frees any `PreparedPrompt` embeddings/position ids in its existing cleanup.

The transformer helper only computes host token/grid metadata. The full
prepared-prompt path still keeps image embeddings, masks, and scatter inputs in
visible `using` declarations.

## Memory / Performance Evidence

- `bun test packages/transformers/src/families/qwen3_5/multimodal/conditional-support.test.ts packages/transformers/src/families/qwen3_5/multimodal/preprocessing.test.ts packages/transformers/src/families/qwen3_5/multimodal/conditional.test.ts packages/serve/src/engine/content.test.ts packages/serve/src/engine/engine.test.ts packages/serve/src/engine/prefix-cache.test.ts`: passed, `96` tests / `383` assertions.
- `bun run typecheck`: passed.
- `bun run lint`: passed.
- `bun test packages/serve`: passed, `351` tests / `1532` assertions.
- `bun run check:coverage`: passed; `@mlxts/serve coverage: 95.02% lines, 95.73% funcs`.
- `bun run validate`: passed.
- `bun run bench:generation -- --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 128 --generation-tokens 16 --trials 1`: passed, `generation_tps=188.317`, `peak_memory=2.651 GB`, `evals_per_token=1.00`.
- `bun run bench:generation:parity -- --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 128 --generation-tokens 16 --trials 1 --skip-mlx-lm-reference`: passed, `generation_tps=195.160` versus stored MLX-LM reference `171.814`, `peak_memory=2.651 GB`, `evals_per_token=1.00`.
- `bun run regression:qwen-image -- --qwen-model mlx-community/Qwen3.6-27B-4bit --report-dir .tmp/qwen-image-prefix-short-circuit`: passed. The first OpenAI Chat image request wrote `92` prompt-cache tokens; exact repeats across OpenAI Chat, OpenResponses, and Anthropic Messages read `92` cached tokens and kept route `single:media_input`.

## Independent Review

Sartre reviewed the next Phase 10 tranche before edits and recommended this
bounded early media prefix-cache short circuit over a persistent visual
embedding cache. The review called out Qwen position ids, expanded token
identity, and `MxArray` ownership as the main risks. This patch keeps expanded
token planning in `@mlxts/transformers`, keeps cache restore family-owned, and
does not retain visual embeddings in serve.

Sartre then reviewed the uncommitted implementation and requested three
tightenings before commit: direct Qwen MRoPE suffix equivalence coverage, a
media-identity guard before skipping prompt preparation, and streaming repeat
coverage. All three are covered in this revision.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Remote and local image transport/decode still run on every request. A separate
host-side decoded-image cache may be worthwhile later, but it needs its own
byte budget and transport-policy review.

General multimodal batching remains out of scope. Media requests still route as
`single:media_input` until prepared-prompt batching semantics are implemented
explicitly.
