# Runtime Review: Serve Cancellation

## Summary

This change adds cooperative request cancellation across `@mlxts/serve` and
`@mlxts/transformers`. HTTP request aborts, SSE client disconnects, and server
shutdown now flow into normalized generation requests, and transformer
generation checks the signal before prefill, between prefill chunks, and between
decode steps.

Non-streaming transformer serving now uses async token-event generation for the
cancellable single-request path, so long requests can observe cancellation
between tokens instead of staying inside one synchronous generate loop.

## Files Reviewed

- `packages/serve/src/batching-engine.ts`
- `packages/serve/src/concurrency-engine.ts`
- `packages/serve/src/errors.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/server-abort.ts`
- `packages/serve/src/server-events.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/types.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/transformers/src/infrastructure/generation/cancellation.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

The new cancellation helper is host-side only and does not allocate `MxArray`
values. Runtime cancellation checks run before prefill, after prefill progress,
before decode token materialization, and after token callbacks. They throw a
typed `GenerationAbortError`, letting existing `finally` blocks dispose sampler
state, prompt embeddings, current/next token handles, streams, caches, and wired
memory limits.

`prefillPromptCache()` still owns and frees sliced prompt embeddings and
position IDs in the same local scope. If cancellation fires after a chunk, the
function exits through the same cleanup path as any other error. Static batched
generation accepts a group-level abort signal for pre-dispatch and cooperative
batch-step checks, but it does not yet support independent per-row abort once a
batch has started.

The serving-layer changes attach and dispose abort-signal listeners around
request scopes. They do not allocate native tensors directly. Queued
concurrency and micro-batch requests reject before dispatch when their signal is
aborted, so they do not mutate model cache state after a client has gone away.

## Memory / Performance Evidence

Validated so far with:

- `bun run typecheck`
- `bun run lint`
- `bun run check:coverage`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:assertions`
- `bun run check:runtime-review`
- `bun test packages/transformers/src/generation.test.ts packages/serve/src/server.test.ts packages/serve/src/concurrency-engine.test.ts packages/serve/src/batching-engine.test.ts packages/serve/src/transformers-engine.test.ts`
- `bun run bench:generation --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 128 --generation-tokens 64 --trials 1 --memory-sample-interval 16`
- `bun run bench:generation:parity --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 128 --generation-tokens 64 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`

The focused tests cover decode-token cancellation, prefill-chunk cancellation,
HTTP signal propagation for completions/chat/Responses, non-streaming
`client_cancelled` error shaping, SSE cancellation propagating into the engine
request signal, queued concurrency aborts, and pending micro-batch aborts.

The local Llama 1B synthetic rung measured `generation_tps=356.258`,
`active_slope_mb_per_token=-0.00`, and `evals_per_token=1.00`. The matching
local parity harness rung measured `generation_tps=358.724`,
`active_slope_mb_per_token=0.01`, and `evals_per_token=1.00`; this run skipped
the external `mlx-lm` half and is not a parity claim against upstream.

Static batch generation remains available for eligible grouped requests.
Cancellable HTTP-backed single requests now trade the synchronous generate
shortcut for cooperative token-event generation, so follow-up serving benchmarks
should measure real endpoint latency separately.

## Independent Review

Sub-agent Kepler completed a read-only cancellation/backpressure audit before
implementation. The audit confirmed that the existing SSE writer could mark a
response cancelled, but transformer generation had no request-level abort
contract. It recommended propagating an abort signal into normalized requests,
checking between prefill chunks and decode steps, rejecting queued work before
dispatch, and avoiding per-row continuous-batching claims.

The implementation follows that tranche and keeps deeper scheduler-level abort,
true pull-driven SSE backpressure, and per-row batch cancellation as follow-up
work rather than pretending this HTTP-layer slice solves continuous batching.

## Remaining Risks / Follow-ups

This is cooperative cancellation, not preemption. A single in-flight MLX forward
or one prefill chunk must finish before the signal is observed, so very large
prefill chunks can still feel sluggish to cancel. The serving defaults should
therefore keep chunk sizes and admission controls honest.

The SSE writer still uses a pump-based writer loop after the stream starts. It
now cancels more reliably and checks before reads, but true pull-driven
backpressure should be a separate reviewed tranche.

Static batch generation has only group-level abort semantics today. The correct
future shape is a scheduler-owned decode loop with per-row abort, memory
reservation, and Qwen/Gemma cache semantics, not a larger HTTP wrapper.
