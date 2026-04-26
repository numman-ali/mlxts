# Runtime Review: Gemma Static Serving Route

## Summary

Gemma 3/4 greedy non-streaming serving can now use the static batch path backed
by `LayerPatternBatchKVCache`. The serving router distinguishes static
eligibility from continuous eligibility: Gemma layer-pattern caches can batch
through `generateBatchTokens()`, but they still cannot enter the full-KV
continuous scheduler.

The transformer engine now has a small static request coalescer for static-only
eligible requests, so concurrent Gemma completions can form a real static batch
instead of silently serializing. Streaming Gemma requests remain single-route
fallbacks with zero continuous scheduler phases.

## Files Reviewed

- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-static.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`

## Tensor Lifetime Audit

This tranche does not add tensor-producing operations or retain `MxArray`
handles directly in `@mlxts/serve`. The new static serving helper only queues
normalized requests, holds the model execution lane, and delegates tensor work
to `generateTransformersBatch()`, which already owns the static generation
scope.

Cancellation handling links request abort signals while a static batch waits for
the model lane and disposes that linked signal in both acquisition-failure and
completion paths. No native arrays are stored in the queue.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run lint`
- `bun run --filter '@mlxts/serve' regression:serve`
- `bun run regression:qwen-gemma -- --profile quick`

The focused serve regression ran 131 tests with 0 failures, and the Qwen/Gemma
quick wrapper passed 67 transformer-focused tests plus the same 131 serve tests.
New coverage proves: Gemma layer-pattern prompts route to `static` with
`generation_batch_start`, concurrent Gemma requests coalesce into a static
batch, Gemma streaming does not emit continuous scheduler phases, and the
real-model regression budget now asserts Gemma static batch rows while keeping
continuous counters at zero.

## Independent Review

Explorer sub-agent Aquinas reviewed the serving integration surface before the
route was widened. The recommendation was to change the route helpers rather
than the transformer batch implementation, add Gemma 3/4 model types only to
static eligibility, split continuous eligibility so Gemma cannot enter the
full-KV scheduler, and update regression evidence to require static batch
counters with zero continuous counters. This implementation follows that shape
and adds the missing concurrent-request coalescer needed for endpoint-level
static batching.

## Remaining Risks / Follow-ups

This is still static greedy batching, not continuous batching. Gemma streaming,
sampled/model-native-default requests, and scheduler row insertion/removal still
fall back until the continuous scheduler can represent layer-pattern cache
extension, filtering, and extraction directly.

The next evidence step is a cached real Gemma endpoint run through
`bun run regression:qwen-gemma -- --profile real` or the serve real-model matrix
when local runtime time is available.
