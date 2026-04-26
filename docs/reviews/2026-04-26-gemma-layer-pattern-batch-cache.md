# Runtime Review: Gemma Layer-Pattern Batch Cache

## Summary

Gemma 3 and Gemma 4 static greedy batch generation now use a transformer-level
batch cache that preserves each layer's attention policy: full-attention layers
retain full KV history, while sliding-attention layers retain only their active
window. The generation layer selects this cache from checkpoint config truth
for Gemma models and leaves serving routes unchanged until endpoint-level
regression proof is added separately.

The proof is intentionally below serving: `generateBatchTokens()` for tiny
Gemma 3 and Gemma 4 models matches separate `generateTokens()` calls with
uneven prompt lengths and sliding-window retention. Gemma 4 coverage includes a
shared-KV layer case.

## Files Reviewed

- `packages/transformers/src/families/gemma3/attention.ts`
- `packages/transformers/src/families/gemma3/block.ts`
- `packages/transformers/src/families/gemma3/model.ts`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/families/gemma4/runtime/attention.ts`
- `packages/transformers/src/infrastructure/cache/index.ts`
- `packages/transformers/src/infrastructure/cache/layer-pattern-batch-state.ts`
- `packages/transformers/src/infrastructure/cache/layer-pattern-batch.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

`LayerPatternBatchKVCache` follows the existing managed-cache ownership model:
append helpers either return owned append results or borrowed cache views, and
public `updateAndFetch()` materializes owned arrays for callers that expect
disposable tensors. Retained cache arrays are exposed through
`retainedLayerStateArrays()`, matching `BatchKVCache`.

Gemma 3 and Gemma 4 batch RoPE offsets are disposable tensors from
`cache.offsetTensor()` and are explicitly freed after query/key RoPE work. Batch
left-padding tensors created for layer-pattern masks are scoped with `using`
inside mask creation, following the existing LLaMA-like left-padded mask
pattern.

`bun run check:tensor-lifetimes` reports no suspicious nested tensor-producing
calls, and `bun run check:assertions` reports no production type assertions.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/infrastructure/masks.test.ts packages/transformers/src/families/gemma3/model.test.ts packages/transformers/src/families/gemma4/model.test.ts packages/transformers/src/families/llama-like/model.test.ts`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run lint`
- `bun run check:file-lines`
- `bun run check:assertions`
- `bun run check:tensor-lifetimes`
- `bun run check:coverage`
- `bun run regression:qwen-gemma -- --profile quick`
- `bun run bench:generation --model google/gemma-3-1b-it --prompt-tokens 128 --generation-tokens 16 --trials 1 --memory-sample-interval 8`
- `bun run bench:generation:parity --model google/gemma-3-1b-it --prompt-tokens 128 --generation-tokens 16 --trials 1 --memory-sample-interval 8 --skip-mlx-lm-reference`

Focused test result: 35 tests passed before the added coverage cases; the full
coverage gate later ran 265 transformer tests and passed with `95.34%` line
coverage / `95.73%` function coverage. The new cache tests prove full and
sliding layers retain different physical lengths after the same prefill, and
that filtering active rows preserves the expected logical length, offsets, and
retained keys. The new mask test covers left-padded sliding masks after stale
positions are trimmed.

Small Gemma 3 1B benchmark evidence:

- `bench:generation`: `prompt_tps=4041.611`, `generation_tps=189.260`,
  `peak_memory=2.181 GB`, `active_slope_mb_per_token=-0.00`,
  `evals_per_token=1.00`.
- `bench:generation:parity`: `prompt_tps=3910.900`,
  `generation_tps=194.613`, `peak_memory=2.181 GB`,
  `active_slope_mb_per_token=-0.00`, `evals_per_token=1.00`. The run skipped
  live mlx-lm capture but printed the stored Gemma 3 1B mlx-lm reference
  (`generation_tps=50.538`, `peak_memory=2.683 GB`, captured 2026-04-05).

This tranche is correctness-first. It does not claim endpoint batching
throughput or continuous batching for Gemma yet.

## Independent Review

Explorer sub-agent Euclid reviewed the smallest correct Gemma batching tranche.
The recommendation was to add a Gemma-only layer-pattern batch cache below
serving, prove `generateBatchTokens()` parity for Gemma 3/4, keep Qwen out of
scope, and leave serving route widening for a later proof. This implementation
follows that shape.

## Remaining Risks / Follow-ups

The ordered sliding batch cache is not yet optimized like the single-request
ring-buffer cache. That is acceptable for this tranche because the keeper bar
was correctness parity first; performance should be measured before any serving
route is widened.

Serving still reports Gemma as ineligible for static/continuous batching. The
next tranche should add endpoint-level route widening only for static greedy
Gemma requests, backed by `regression:qwen-gemma` route evidence. Continuous
Gemma batching remains separate scheduler work because row extension/removal
must respect layer-pattern cache state.
