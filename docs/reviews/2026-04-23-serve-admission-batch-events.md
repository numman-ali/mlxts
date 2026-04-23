# Runtime Review: Serve Admission Batch Events

## Summary

Added an observable `generation_admission_batch` event for the serving
micro-batching queue. Operators can now distinguish requests that were admitted
and coalesced by the HTTP-level queue from requests that later reached the
transformer engine's real static batch path.

The CLI logs admission batches by default with the batch size, model, request
ids, per-request max token limits, and whether the inner engine will receive a
batch call or sequential fallback.

## Files Reviewed

- `packages/serve/src/batching-engine.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/types.ts`

## Tensor Lifetime Audit

No tensor, MLX, cache, or native-resource code changed. This only emits metadata
about already-normalized generation requests before the existing micro-batch
settlement path calls the inner generation engine.

## Memory / Performance Evidence

Validation used fake engines and tiny test models only; no large live model
serving runs were started.

- `bun test packages/serve/src/batching-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/src/cli.test.ts`
- `bun test packages/serve/src`
- `bun run lint`
- `bun run typecheck`
- `bun run check:assertions`
- `bun run check:file-lines`

## Independent Review

Dalton's serving audit identified batching truthfulness as a remaining
observability gap: static batch starts were visible, but the admission
micro-batching layer had no event surface and fallback behavior was opaque. This
change keeps the distinction explicit instead of calling admission
micro-batching continuous batching.

## Remaining Risks / Follow-ups

Admission batches are emitted only for batches larger than one request to avoid
turning normal single-request serving into log noise. Continuous token-level
batching still requires a scheduler/cache-aware engine below `@mlxts/serve`.
