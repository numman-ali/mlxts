# Runtime Review: Serve chat streaming and admission control

## Summary

This follow-up moves `@mlxts/serve` from a mostly one-shot model server toward a
safer local serving surface for real Gemma/Qwen usage. The change has two
pieces that belong together.

First, `@mlxts/transformers` now exposes async token-generation events from the
real decode loop. `@mlxts/serve` uses that to back `/v1/completions` and
`/v1/chat/completions` streaming from the actual model path instead of only from
stub engines. Chat SSE formatting now keeps Qwen-style reasoning in
`reasoning_content` deltas instead of leaking raw `<think>` tags into visible
assistant text.

Second, single-model serving now has an explicit concurrency gate. Admission
micro-batching is still useful for nearby non-streaming requests, but it was not
enough once streaming entered the default path because streaming requests could
otherwise overlap the same local model runtime. The new bounded in-flight gate
serializes `generate`, `generateBatch`, and `stream` work for one served model
instance by default.

## Files Reviewed

- `packages/transformers/src/types.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`
- `packages/transformers/src/index.ts`
- `packages/serve/src/concurrency-engine.ts`
- `packages/serve/src/protocols/openai-stop.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/protocols/openai-chat-completion-streaming.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-completions.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/index.ts`

## Tensor Lifetime Audit

The runtime-sensitive part of this change is the new async token-event path now
split across `packages/transformers/src/infrastructure/generation/runtime.ts`
and `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`.
I re-checked the same ownership boundaries as the synchronous generation loop:

- prompt-side retained embeddings and position ids are still freed exactly once
- scheduled async lookahead tokens still have a clear owner and are released on
  both normal completion and early generator teardown
- prompt caches created internally remain owned by the streaming generator and
  are disposed when the async iterable completes or is cancelled
- the dedicated generation stream and temporary wired-limit override now live
  for the full async-iteration lifetime and are restored in a finalizer path

The serving-side additions are host-only queueing, SSE formatting, and request
normalization code. They do not create new native tensor owners beyond the model
generation path above.

## Memory / Performance Evidence

Validated with:

- `bun run typecheck`
- `bun test packages/transformers/src/generation.test.ts packages/serve/src/concurrency-engine.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server.test.ts packages/serve/src/model-server.test.ts packages/serve/src/cli.test.ts`
- `bun run bench:generation` and `bun run bench:generation:parity` were not rerun for this slice because the change does not claim a decode-throughput win or model-math parity shift; it changes streaming ownership, SSE formatting, and admission control around the existing decode loop rather than the underlying family math or benchmark target configuration.

Added focused coverage for:

- async token-event generation summaries in `packages/transformers/src/generation.test.ts`
- real transformers-engine streaming for text prompts and prompt-open thinking
- bounded in-flight admission across `generate` and `stream`
- OpenAI chat streaming normalization, SSE chunk formatting, and reasoning split
- end-to-end chat SSE behavior in the Bun fetch handler

The performance posture is intentionally conservative: streaming now rides the
real decode loop, but we still keep the first serving scheduler simple. The
concurrency gate prevents overlapping local decode loops on one model instance,
which is the right safety move before deeper continuous-batching work lands.

## Independent Review

Independent sub-agent audits informed the shape of this change.

- `Beauvoir` audited `vllm-mlx`, `oMLX`, `mlx-lm`, and TGI and confirmed the
  durable upstream pattern: keep SSE/wire formatting thin, keep one scheduler or
  one engine loop as the owner of model progress, and validate concurrency at
  admission rather than trusting async HTTP handlers.
- `Lorentz` audited the current repo and confirmed the most important remaining
  serving gaps after admission micro-batching: default model-backed streaming
  was still missing, chat streaming was explicitly unsupported, and one model
  instance still lacked a true cross-API concurrency guard.

Those findings are reflected directly here: this patch adds model-backed
streaming and a shared in-flight gate, but it does not pretend that admission
micro-batching alone is continuous batching.

## Remaining Risks / Follow-ups

- This is still not a continuous-batching scheduler. Nearby non-streaming
  requests can coalesce, but the next real throughput step is a waiting/running
  request scheduler with chunked prefill and shared decode ownership.
- Streaming cancellation/backpressure is better behaved because async iterables
  now own cleanup, but the server still does not expose a richer cancellation API
  or request-level metrics surface.
- Structured OpenAI `tool_calls` output is still not emitted from the serve
  layer. Tool schemas are accepted, but standard structured tool-call responses
  remain a follow-up.
- Multi-model lifecycle management is still separate from this slice. The router
  can multiplex prebuilt engines, but model pooling and memory-aware multi-model
  serve remain future work.
