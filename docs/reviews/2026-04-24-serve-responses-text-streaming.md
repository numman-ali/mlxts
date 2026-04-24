# Runtime Review: Serve Responses Text Streaming

## Summary

`@mlxts/serve` now treats `/v1/responses` as a useful text endpoint instead of
a non-streaming shim. The adapter accepts OpenAI-style string input or text-only
message item arrays, preserves model-native sampling defaults, supports stop
sequences and the local Qwen thinking template controls, and emits semantic SSE
events for streamed text and reasoning output.

Unsupported Responses features remain explicit rejections: persisted state,
background jobs, tools, files/images/audio, prompt templates, structured output,
and request-side reasoning controls. This keeps the protocol adapter thin while
making the text path usable by real Responses clients.

## Files Reviewed

- `packages/serve/src/protocols/openai-responses.ts`
- `packages/serve/src/protocols/openai-responses-formatting.ts`
- `packages/serve/src/protocols/openai-responses-input.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

The changed production files are protocol parsing, JSON/SSE formatting, and HTTP
route orchestration. They do not create tensors, call MLX operations, mutate KV
caches, allocate native handles, or change transformer generation internals.
Streaming cancellation still flows through the normalized request abort signal
before reaching the transformer engine.

## Protocol Evidence

The implementation was checked against official OpenAI Responses documentation:
`input` accepts either a string or an array of response input items, `stream:
true` emits typed server-sent events, and the text stream event family includes
`response.created`, `response.in_progress`, `response.output_item.added`,
`response.content_part.added`, `response.output_text.delta`,
`response.output_text.done`, `response.output_item.done`, and terminal
`response.completed` / `response.incomplete` events.

References:

- https://platform.openai.com/docs/guides/streaming-responses
- https://platform.openai.com/docs/api-reference/responses-streaming/response/output_text
- https://platform.openai.com/docs/guides/responses-vs-chat-completions

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/server.test.ts`: 29 pass, 0 fail.
- `bun run typecheck`: pass.
- `bun run check:coverage`: pass.
- `bun run validate`: pass, including `@mlxts/serve` coverage at 95.72% lines / 97.07% functions.

## Independent Review

Zeno audited the pre-change Responses surface and found the same highest-value
gap: text message-array input and semantic SSE streaming were missing, while
tools, persistence, multimodal input, prompt templates, and structured output
should remain explicitly unsupported until implemented for real. The patch
follows that recommended tranche.

## Remaining Risks / Follow-ups

Responses streaming now covers text and reasoning deltas, but not tool calls or
stateful conversation continuation. The next protocol tranches should be
Responses function/tool call parity and Anthropic Messages after serving
benchmark quality is back on track. Serving architecture still needs a
scheduler-owned continuous batching engine before Qwen/Gemma concurrency can be
claimed as true active-row batching instead of admission queueing.
