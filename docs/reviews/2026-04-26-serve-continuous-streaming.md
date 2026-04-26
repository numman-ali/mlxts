# Runtime Review: Serve Continuous Streaming

## Summary

Streaming requests can now use the existing full-KV continuous scheduler when
they are greedy, batch-eligible, and served by a full-cache LLaMA-like model.
The stream path shares the same text-delta decoding helper as single-request
streaming, emits the same scheduler phase telemetry as buffered continuous
generation, and aborts scheduler work when the consumer closes the iterator.

This does not broaden Qwen or Gemma claims. Both remain single-lane fallbacks
until their hybrid and layer-pattern cache semantics are implemented below the
serving layer.

## Files Reviewed

- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/src/transformers-engine-streaming.ts`
- `packages/serve/src/transformers-engine.ts`

## Tensor Lifetime Audit

No new MLX tensor-producing operations were added. The continuous stream path
uses the existing `ContinuousBatchTokenScheduler`, `BatchKVCache`, and model
forward paths. The new queue is JS-only and stores protocol stream events, text
fragments, and completion metadata.

Iterator cleanup now matters more because streaming can be scheduler-backed.
The stream generator links the request abort signal to a request-local abort
controller, aborts if the consumer stops before scheduler completion, waits for
the scheduler promise to settle, disposes the linked signal, and removes the
request id from scheduler metadata.

`server-streaming.ts` and `server-responses-streaming.ts` now call
`iterator.return()` when stop-sequence filtering ends a stream early, so model
iterators can release caches and scheduler rows instead of continuing hidden
decode work.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/serve/src/transformers-engine.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun test packages/serve/src/server-streaming.test.ts packages/serve/src/transformers-engine.test.ts`
- `bun run check:coverage`
- `bun run regression:qwen-gemma -- --profile quick`

This change removes duplicate stream text-delta logic from
`transformers-engine.ts` and reuses it from `transformers-engine-streaming.ts`.
For eligible streamed requests, model execution moves from the serial model lane
to the scheduler lane, so concurrent greedy streams can batch at the decode
step. For ineligible models and sampled requests, routing still falls back to
the single model lane.

## Independent Review

Two explorer sub-agents reviewed the deeper Qwen and Gemma batching questions.
Both concluded that Qwen/Gemma should remain serving fallbacks until transformer
cache semantics are upgraded: Qwen needs a hybrid batch cache with masked
linear-attention state, while Gemma needs per-layer full/sliding batch cache
state and per-layer padding/mask semantics.

Their review supports keeping this tranche limited to the already-proven
full-KV scheduler, rather than pretending the serve layer can route Qwen/Gemma
before the lower cache contract is correct.

## Remaining Risks / Follow-ups

The continuous streaming path is still greedy-only and full-KV-only. Stop
sequence handling closes the iterator now, but the final protocol finish reason
is still determined by the SSE adapter's stop filter, not by a scheduler-native
stop token.

The next cache-semantics work should be transformers-first: Qwen static greedy
batch proof with masked recurrent state, and Gemma static greedy batch proof
with per-layer sliding/full cache padding. Only after those match separate
single-request generation should serving routes be widened.
