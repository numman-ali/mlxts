# Serve Stream Writer Scaffolding Review

## Summary

Extracted the shared SSE writer loop into `packages/serve/src/streaming/writer-base.ts`.
The helper owns headers, iterator normalization, heartbeat wrapping, cancellation
reads, HTTP-yield timing, and early iterator return. OpenAI completions, OpenAI
chat, OpenResponses, and Anthropic Messages keep their protocol-specific state
machines, terminal chunks, usage chunks, reasoning parsing, stop filtering, and
tool-call handling.

## Files Reviewed

- `packages/serve/src/http/routes-anthropic.ts`
- `packages/serve/src/http/routes-responses.ts`
- `packages/serve/src/http/server.ts`
- `packages/serve/src/streaming/writer-anthropic-messages.ts`
- `packages/serve/src/streaming/writer-base.ts`
- `packages/serve/src/streaming/writer-openai-responses.ts`
- `packages/serve/src/streaming/writer-openai.ts`

## Tensor Lifetime Audit

No tensor-producing expressions, model execution paths, cache ownership,
generation schedulers, media preparation, or protocol formatting state machines
changed. The production change is limited to stream-control scaffolding and
import rewiring.

## Memory / Performance Evidence

This tranche makes no performance claim. `bench:generation` and
`bench:generation:parity` were not run because no model forward, sampler,
cache, or scheduler hot path changed.

Focused validation passed:

- `bun test packages/serve/src/streaming/writer-base.test.ts packages/serve/src/streaming/runtime.test.ts packages/serve/src/streaming/writer-openai.test.ts packages/serve/src/http/server.test.ts packages/serve/src/protocols/openai-completions.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts` passed: 105 tests.
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run validate`

## Independent Review

Mendel independently reviewed the dirty tree and recommended the same helper
shape: one shared loop for `toAsyncIterator`, `withSseHeartbeat`,
`readStreamEvent`, handler dispatch, `yieldToHttpWriter`, and early
`iterator.return()`, with finalization staying outside the helper. The review
called out the same risk boundary preserved here: do not move stop filtering,
reasoning parsing, tool-call parsing, terminal chunks, usage chunks, or
Anthropic content-block state into the shared scaffold.

## Remaining Risks / Follow-ups

Responses and Anthropic still have similar named-event SSE helpers. They are
small protocol-adjacent framing helpers and were left in place to keep this
tranche focused on the shared stream-control loop.
