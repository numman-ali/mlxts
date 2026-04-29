# Continuous Prefix-Cache Seeding

## Summary

This tranche widens prompt-prefix reuse from the single-request lane into the
continuous scheduler without adding a paged cache backend. Serve still owns
prompt matching, identity checks, eviction, usage accounting, and telemetry.
Transformers owns restoring family cache state into a one-row batch cache.

Paged KV, block deduplication, CoW, SSD tiers, and quantized KV remain separate
cache-backend work. This tranche uses the existing managed cache contracts only.

## Files Reviewed

- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/infrastructure/cache/batch-state.ts`
- `packages/transformers/src/infrastructure/cache/batch.ts`
- `packages/transformers/src/infrastructure/cache/layer-pattern-batch.ts`
- `packages/transformers/src/infrastructure/generation/batch-cache-factory.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-lifecycle.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-prefill.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/families/qwen3_5/cache/index.ts`
- `packages/transformers/src/families/qwen3_5/cache/batch-cache.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/generation.ts`
- `packages/serve/src/engine/continuous.ts`
- `packages/serve/src/engine/index.ts`

## Runtime Sensitivity Notes

The changed path is runtime-sensitive because it changes cache ownership and
continuous scheduler admission for message prompts. A prompt-cache hit now
passes an owned single-cache fork into the scheduler with the full logical
prompt token history and the cached prefix length. The scheduler restores that
state into a one-row batch cache, disposes the transferred single cache, and
prefills only the uncached suffix before joining the active decode batch.

Prompt accounting, admission budgets, sampler history, repetition penalty, stop
handling, and usage fields continue to see the full prompt. Suffix-only tokens
are used only for model prefill work after the restored cache offset.

## Tensor Lifetime Audit

Seeded restore copies cache state into a fresh batch cache before the source
fork is disposed. Full KV and layer-pattern caches move cloned layer snapshots
into batch-owned layer state and null the snapshot handles before disposal.
Layer-pattern restore preserves the sliding cursor so exact continuation uses
the same ring-buffer ordering as the single-cache snapshot.

Qwen restore copies full-attention layer snapshots through the managed
`BatchKVCache` layer restore path and copies linear recurrent/conv state through
the Qwen cache's family-owned state API. Serve never sees KV tensors, layer
arrays, Qwen recurrent state, or Gemma sliding-window internals.

Continuous queued requests dispose transferred prefix caches if validation
fails, if a zero-token request returns before scheduling, or if enqueue sees an
already-aborted signal. Waiting/prefilling cleanup disposes unconsumed prefix
caches, and prefilling consumes then deletes the transferred cache after
successful batch restore. Prompt-boundary snapshots produced from prefilling
rows follow the same callback ownership rule as single-request generation: the
callback owns the snapshot, and callback failure disposes it before propagating
the error.

Cold message requests with prompt-cache snapshot callbacks stay on the batched
initial prefill path. When a cold full batch needs snapshots, the scheduler
batches the reusable prompt prefix, emits one snapshot per extracted row at the
`prompt.length - 1` boundary, then batches the final prompt-token forward that
samples the first generated token. Seeded prefix-hit rows still use the
prefilling lane because they restore a request-local one-row batch cache before
prefilling only the suffix.

## Memory / Performance Evidence

- `bun run validate`: passed.
- `bun test packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/qwen3_5/cache/cache.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`: passed, `134` tests.
- `bun run regression:qwen-gemma -- --profile real`: passed.
- Transformer decode smoke inside the real regression:
  - Qwen3.6 27B 4-bit: `prompt_tps=250.578`, `generation_tps=29.135`, `evals_per_token=1.00`, `peak_memory=17.184`.
  - Gemma 4 E2B: `prompt_tps=8154.054`, `generation_tps=82.498`, `evals_per_token=1.00`, `peak_memory=9.893`.
- Protocol prompt-cache evidence inside the real regression:
  - Qwen chat/responses: `cache_read_tokens=139`, `prompt_cache_hits=1`, `prompt_cache_read_tokens=278`, `routes=continuous:eligible=1`.
  - Qwen Anthropic: `prompt_cache_hits=1`, `prompt_cache_read_tokens=278`, `routes=continuous:eligible=1`.
  - Gemma chat/responses: `cache_read_tokens=138`, `prompt_cache_hits=1`, `prompt_cache_read_tokens=276`, `routes=continuous:eligible=1`.
  - Gemma Anthropic: `prompt_cache_hits=1`, `prompt_cache_read_tokens=276`, `routes=continuous:eligible=1`.
- Long/short fairness evidence inside the real regression:
  - Qwen `32768x128+128x32` staggered stream passed budgets with `continuous_admissions=2`, `routes=continuous:eligible=2`, `mean_server_prefill_tps=122.618`, `mean_post_ttft_completion_tps=13.938`, `peak_memory=19.277`.
  - Gemma `5000x128+128x32` staggered stream passed budgets with `continuous_admissions=2`, `routes=continuous:eligible=2`, `mean_server_prefill_tps=3098.266`, `mean_post_ttft_completion_tps=68.251`, `peak_memory=9.841`.
- `bench:generation`: not rerun separately for this serving-cache tranche; no model math or decode kernel changed.
- `bench:generation:parity`: covered by the real Qwen/Gemma regression decode smoke above.

## Independent Review

Einstein reviewed the cache and scheduler shape before implementation. The
recommendation was to avoid paged KV in this tranche and implement
continuous-scheduler prefix-hit seeding over existing managed cache contracts.
Full-KV snapshots can use arbitrary LCP only when trimmable; Qwen hybrid and
Gemma layer-pattern/sliding snapshots remain exact-boundary unless
`snapshot.canFork()` says otherwise; content/media prompts stay on the single
lane because continuous batching does not carry prepared embeddings or position
ids.

Helmholtz reviewed the uncommitted diff after implementation. The follow-up
risks were transferred-cache disposal on synchronous enqueue exits, cold
snapshot callbacks forcing one-row prefill, and seeded restore failures
becoming scheduler-wide. The current diff adds focused tests and fixes for all
three.

## Out-of-scope Drift Noticed

Explicit `generateBatch()` with `maxBatchSize <= 1` can still use static
batching for message-shaped requests and does not participate in prompt-prefix
reuse. That behavior predates this tranche; the new route covers ordinary
`generate()` and streaming requests when continuous scheduling is enabled.

## Remaining Risks / Follow-ups

This is not paged KV and does not deduplicate cache storage across active rows.
It restores a prefix-hit snapshot into one request-local batch row, then merges
that row into the managed continuous batch cache.

The real Qwen/Gemma regression should remain the acceptance guard for endpoint
behavior: prompt-cache read evidence must stay nonzero, and uncached continuous
fairness/high-concurrency rungs must not regress.
