# Runtime Review: Static Batch Generation Foundation

## Summary

Added the first real batch-generation foundation below serving: a managed `BatchKVCache`,
left-padded batch masks, LLaMA-like full-cache batch forward support, and greedy
static `generateBatchTokens()`. This is intentionally not continuous batching yet;
it proves the lower transformer/cache layer can run a fixed active batch before the
serving scheduler owns request admission and decode steps.

## Files Reviewed

- `packages/transformers/src/families/gemma3/model.ts`
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/families/llama-like/attention.ts`
- `packages/transformers/src/families/llama-like/block.ts`
- `packages/transformers/src/families/llama-like/model.ts`
- `packages/transformers/src/families/qwen3_5/conditional.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/cache/batch.ts`
- `packages/transformers/src/infrastructure/cache/index.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/infrastructure/cache/view.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

Cache state remains explicitly disposable. `BatchKVCache.arrays()` returns retained
visible state arrays owned by callers, matching the existing single-cache contract.
Metadata tensors from `offsetTensor()` and `leftPaddingTensor()` are caller-owned
and are held with `using` at call sites.

The left-padded mask helper keeps all tensor-producing intermediates in named
`using` bindings. `generateBatchTokensInternal()` frees sampled token tensors on
each decode step and owns its internal `BatchKVCache` inside the generation scope.

## Memory / Performance Evidence

- `bun test packages/transformers/src/families/llama-like/model.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/infrastructure/masks.test.ts packages/transformers/src/generation.test.ts`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`
- `bun run bench:generation --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 32 --generation-tokens 8 --trials 1`
  - prompt_tps=1608.491, generation_tps=191.520, peak_memory=2.547 GB, active_slope_mb_per_token=-0.00, evals_per_token=1.00
- `bun run bench:generation:parity --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 32 --generation-tokens 8 --trials 1`
  - prompt_tps=1431.311, generation_tps=208.925, peak_memory=2.547 GB, active_slope_mb_per_token=-0.00, evals_per_token=1.00
  - live `mlx-lm` reference capture was unavailable locally because `mlx_lm` is not installed; the script used the recorded baseline reference for comparison.

The new parity test compares static greedy batch generation against separate
single-prompt greedy generation on the same LLaMA-like model. No long soak or
full baseline-shaped benchmark run was performed for this tranche because it is
a small-cache foundation and not yet wired into serving. The small 32/8 benchmark
probes are sanity checks for eval count and memory slope, not comparable to the
1024/128 recorded prompt-throughput baselines.

## Independent Review

Laplace independently audited the transformer batch-cache tranche and identified
the original `BatchKVCache.extend()` capacity-padding bug plus the missing model
and mask integration needed before real batching. Mencius independently audited
the serving layer and confirmed the correct sequencing: land transformer
batch-cache/session primitives first, then replace serving with one scheduler-owned
engine rather than adding another admission wrapper.

## Remaining Risks / Follow-ups

`generateBatchTokens()` is intentionally static, full-cache, non-streaming, and
greedy-only. Qwen hybrid caches, Gemma 3/4 layer-pattern caches, sliding-window
batch caches, sampled batched decoding, and continuous serving scheduling remain
future tranches.

The next serving tranche should build one scheduler-owned engine around this lower
batch primitive, with request queues, cancellation, active-row filtering, and
streaming output collectors owned by that engine.
