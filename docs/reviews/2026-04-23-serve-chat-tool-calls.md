# Runtime Review: Serve Chat Tool Calls

## Summary

Added non-streaming structured OpenAI chat `tool_calls` output for generated
tool-call envelopes. When a chat request includes tools, valid generated
`<tool_call>...</tool_call>` blocks are stripped from assistant content and
returned as OpenAI-compatible `message.tool_calls` with
`finish_reason: "tool_calls"`.

The parser supports both the JSON envelope used by the local agent instructions
and Qwen-style native function blocks. Streaming tool-call deltas are left as a
separate follow-up because the current SSE path emits visible content
incrementally and should not risk leaking partial XML before a buffering design
exists.

## Files Reviewed

- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-chat-tool-calls.ts`

## Tensor Lifetime Audit

This change is protocol formatting only. It parses generated text on the host
after generation has completed and does not allocate, retain, or dispose MLX
tensors.

No generation engine, cache, streaming iterator, or model-forward code changed.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server.test.ts packages/agent/src/chat-model.test.ts`
- `bun run typecheck`
- `bun run lint`

The tests prove:

- JSON `<tool_call>` envelopes format as OpenAI `message.tool_calls`
- Qwen-style `<function=name>` tool calls format into JSON-string arguments
- reasoning plus multiple tool calls keeps `reasoning_content` and does not leak
  XML into visible content
- malformed generated tool-call text remains visible content instead of crashing
  the server
- tool-call-looking text is not parsed when the request did not enable tools

No throughput benchmark is required because this is response formatting around
completed generation, not a decode hot-path change.

## Independent Review

Feynman independently audited the gap and recommended this exact scope:
implement non-streaming structured `tool_calls` first, keep the parser local to
`@mlxts/serve` instead of importing `@mlxts/agent`, and leave streaming tool-call
deltas for a separate buffered SSE tranche.

## Remaining Risks / Follow-ups

Streaming chat responses still do not emit OpenAI `delta.tool_calls`. That
should be added only after the streaming path can buffer enough text to detect
tool-call envelopes without leaking partial XML as normal content.

This change parses tool-call envelopes; it does not execute tools. Tool
execution remains the responsibility of `@mlxts/agent` or external OpenAI-
compatible clients.
