# Runtime Review: Serve Chat Tool Calls

## Summary

Added structured OpenAI chat `tool_calls` output for generated tool-call
envelopes. When a chat request includes tools, valid generated
`<tool_call>...</tool_call>` blocks are stripped from assistant content and
returned as OpenAI-compatible non-streaming `message.tool_calls` or streaming
`delta.tool_calls`, with `finish_reason: "tool_calls"`.

The parser supports both the JSON envelope used by the local agent instructions
and Qwen-style native function blocks. The streaming path buffers until a full
tool-call envelope is available, so split XML never leaks as visible assistant
content. Malformed or incomplete tool-call-looking text is preserved as content
instead of being swallowed or crashing the server.

## Files Reviewed

- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-chat-completion-streaming.ts`
- `packages/serve/src/protocols/openai-chat-tool-call-stream.ts`
- `packages/serve/src/protocols/openai-chat-tool-calls.ts`
- `packages/serve/src/server-stop-filter.ts`
- `packages/serve/src/server-streaming.ts`

## Tensor Lifetime Audit

This change is protocol/SSE formatting only. It parses generated text on the host
after tokens have been decoded and does not allocate, retain, or dispose MLX
tensors.

No model-forward, cache, tokenizer, or MLX array lifetime code changed. The
streaming iterator wrapper now holds short host strings while detecting complete
tool-call envelopes.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server-streaming.test.ts packages/serve/src/server.test.ts`
- `bun run typecheck`
- `bun run lint`
- `bun run check:assertions`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`
- `bun run build`

The tests prove:

- JSON `<tool_call>` envelopes format as OpenAI `message.tool_calls`
- Qwen-style `<function=name>` tool calls format into JSON-string arguments
- reasoning plus multiple tool calls keeps `reasoning_content` and does not leak
  XML into visible content
- malformed generated tool-call text remains visible content instead of crashing
  the server
- tool-call-looking text is not parsed when the request did not enable tools
- split streaming Qwen-style tool-call XML emits `delta.tool_calls` without XML
  leakage
- streaming reasoning remains in `reasoning_content` while multiple tool calls
  keep stable indexes and ids

No throughput benchmark is required because this is host-side response/SSE
formatting around decoded text, not a decode hot-path change.

## Independent Review

Feynman independently audited the gap and recommended this exact scope:
implement non-streaming structured `tool_calls` first, keep the parser local to
`@mlxts/serve` instead of importing `@mlxts/agent`, and leave streaming tool-call
deltas for a separate buffered SSE tranche.

Bernoulli independently audited the streaming tranche against OpenAI and TGI
shapes. The recommendation was to emit `choices[0].delta.tool_calls`, preserve
indexes, use `finish_reason: "tool_calls"`, only parse when tools are enabled,
and buffer XML/Qwen-style function blocks so partial tool-call text does not leak
as assistant content.

## Remaining Risks / Follow-ups

The streaming implementation emits complete tool-call argument strings once the
full envelope is parsed rather than fragmenting arguments token by token. This is
OpenAI-compatible for clients that accumulate deltas, but not yet byte-for-byte
identical to providers that stream each argument fragment as it is generated.

This change parses tool-call envelopes; it does not execute tools. Tool
execution remains the responsibility of `@mlxts/agent` or external OpenAI-
compatible clients.
