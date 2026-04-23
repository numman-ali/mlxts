# Runtime Review: Static Batch Decode V2

## Summary

Made the static greedy batch primitive row-aware without widening it into
continuous batching. `generateBatchTokens()` now accepts either one scalar
`maxTokens` value for the whole batch or a per-row `maxTokens` array, emits
row-indexed token/done events, and skips zero-token rows before model prefill.

`@mlxts/serve` now uses that lower primitive for eligible mixed-`max_tokens`
non-streaming greedy full-cache groups. Qwen hybrid caches, Gemma 3/4
layer-pattern caches, sampled/model-native-default requests, and streaming still
fall back to single-request generation until deeper cache/scheduler work lands.

## Files Reviewed

- `packages/transformers/src/types.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/transformers-engine-generation.ts`

## Tensor Lifetime Audit

The changed batch loop keeps the previous explicit ownership structure.
`currentToken` is freed exactly once per decode step, token extraction still uses
short-lived `slice(...)` views inside `using`, and `nextInput` remains a named
`using` binding. The internal `BatchKVCache` is still owned by
`generateBatchTokensInternal()` inside `runGenerationScope()`.

Per-row completion only changes which active rows continue. Finished rows are
not passed to `cache.filter()`, and the loop breaks before filtering when every
active row has finished. Zero-token rows are excluded from the initial active
batch so they do not force prefill/cache allocation work.

## Memory / Performance Evidence

Validated with:

- `bun test packages/transformers/src/generation.test.ts packages/transformers/src/families/llama-like/model.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/src/cli.test.ts`
- `bun run typecheck`
- `bun run bench:generation --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 32 --generation-tokens 8 --trials 1`
  - prompt_tps=1652.434, generation_tps=195.532, peak_memory=2.547 GB, active_slope_mb_per_token=-0.00, evals_per_token=1.00
- `bun run bench:generation:parity --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 32 --generation-tokens 8 --trials 1`
  - prompt_tps=1466.953, generation_tps=209.503, peak_memory=2.547 GB, active_slope_mb_per_token=-0.00, evals_per_token=1.00
  - live `mlx-lm` reference capture was unavailable locally because `mlx_lm` is not installed; the script used the recorded baseline reference for comparison.

The small 32/8 probes are hot-path sanity checks, not a full throughput claim.
Both retain flat active memory and one blocking eval per token. The prompt TPS
warning matches the known small-prompt baseline mismatch pattern recorded in the
previous static batch review and is not caused by this batch-only row-limit
change.

The new tests prove scalar batch lengths remain unchanged, per-row lengths
preserve original result order, zero-token rows do not enter the active prefill
batch, invalid row-length arrays fail early, mixed-`max_tokens` serving batches
use the static batch path, and concurrent HTTP completions with mixed
`max_tokens` coalesce through `serveLoadedModel()`.

## Independent Review

Ptolemy audited the serving/batching state against `mlx-lm`, `vllm-mlx`, and
`omlx` and recommended this exact next tranche: make the static batch path
scheduler-shaped with per-row generation lengths and token-event output before
attempting Qwen/Gemma batching.

Arendt independently reviewed the local primitive and serving integration. That
review called out the main traps: active batch position vs original batch index,
not filtering an empty active set, excluding zero-token rows from prefill, and
normalizing per-row max tokens before passing a scalar maximum through defaults
resolution. The implementation follows those constraints.

## Remaining Risks / Follow-ups

This is still static greedy full-cache batching, not continuous batching.
Streaming over the new row-indexed event seam, cancellation, and request
lifecycle scheduling are future serving-engine work.

Gemma 3/4 layer-pattern/sliding caches and Qwen hybrid cache state still need
separate batch-cache designs before they can safely use this path. Sampled
batched decode also remains out of scope until sampler state is made row-aware.
