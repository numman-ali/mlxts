# OpenResponses Streaming Function Tools

## Summary

This tranche removes the temporary `stream: true` rejection for active
OpenResponses function tools. `/v1/responses` now streams generated function
calls as semantic Responses SSE events while preserving the existing text,
reasoning, stop-sequence, cancellation, and stream-observability paths.

The implementation remains bounded to model-generated function-call output.
It does not execute tools, add built-in tools, widen forced tool choice, add
stateful continuation, or accept rich function outputs.

## Files Reviewed

- `packages/serve/src/protocols/openai-responses.ts`
- `packages/serve/src/streaming/writer-openai-responses.ts`
- `packages/serve/src/streaming/writer-openai-responses-tools.ts`
- `packages/serve/src/streaming/writer-openai-responses.test.ts`
- `packages/serve/src/protocols/openai-responses.test.ts`
- `packages/serve/src/http/server.test.ts`

## External Contract Check

- OpenAI's function-calling guide describes streaming Responses function calls
  as `response.output_item.added`, followed by
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and `response.output_item.done`.
  Source: https://developers.openai.com/api/docs/guides/function-calling
- The Responses streaming API reference names
  `response.function_call_arguments.delta` as the partial function-call
  argument event. Source:
  https://developers.openai.com/api/reference/resources/responses/methods/create

## Runtime Review

`writer-openai-responses.ts` now reuses the conservative generated tool-call
stream parser already used by OpenAI Chat streaming. The parser is only enabled
when active function tools reached the normalized `GenerationInput`; when tools
are inactive or `tool_choice: "none"` suppressed them, tool-looking output
remains visible text.

`writer-openai-responses-tools.ts` owns the Responses-specific SSE event shape
for function calls. It emits the function-call output item, one argument delta
for the completed parsed argument string, the argument-done event, and the
completed output item. The terminal `response.completed` object is rebuilt from
the streamed output state so streamed function-call items match the final
response object.

No model execution, cache layout, scheduler policy, sampling, tensor lifetime,
or MLX native path changed.

## Tensor Lifetime Audit

No tensor-producing code changed. The changed production files operate on HTTP
request metadata, normalized request shapes, strings, and SSE JSON payloads.

## Memory / Performance Evidence

- The tranche does not add per-token model work. Function-call parsing is
  string-side stream buffering already used by Chat Completions.
- Text-only Responses streaming still follows the existing reasoning and stop
  filtering path.
- Active function tools hold generated tool-call envelope text until a complete
  conservative envelope is parsed, matching the existing chat-streaming
  behavior and avoiding malformed partial tool events.

## Independent Review

Kant recommended this exact tranche as the next bounded Phase 9 product move
after non-streaming OpenResponses function tools. The review called out two
risks that this implementation addresses directly: use Responses SSE event
names rather than Chat Completions chunks, and keep malformed/tool-looking text
visible when tools are inactive.

## Validation

- `bun test packages/serve/src/streaming/writer-openai-responses.test.ts packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/http/server.test.ts`
  - `59 pass`
- `bun run --filter '@mlxts/serve' typecheck`
  - passed
- `bun run check:file-lines`
  - passed
- `bun run lint`
  - passed
- `bun run check:runtime-review`
  - passed
- `bun run check:assertions && bun run check:per-package-agents && bun run check:cross-package-imports && bun run check:tensor-lifetimes`
  - passed
- `bun run check:coverage`
  - passed; `@mlxts/serve` coverage reported `95.03%` lines and `95.79%`
    functions
- `bun run validate`
  - passed

## Remaining Risks / Follow-ups

- Existing Responses text and reasoning SSE events predate this tranche and
  keep their current payload shape. This tranche adds `response_id` to the new
  function-call events it owns; a later protocol-compliance pass can audit
  `response_id` coverage across all Responses SSE events.

## Out-of-scope Drift Noticed

- No unrelated drift was changed.
