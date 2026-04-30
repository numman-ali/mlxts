# Anthropic Messages Tool Use

## Summary

Added bounded Anthropic Messages tool-use support in `@mlxts/serve`. The `/v1/messages` adapter now accepts client tool definitions, assistant `tool_use` history blocks, and user `tool_result` history blocks, then maps them through the protocol-neutral `ChatTool` / `ChatToolCall` contracts used by the serving engine. Non-streaming generated tool-call envelopes format back to Anthropic `tool_use` content blocks with `stop_reason: "tool_use"`.

## Files Reviewed

- `packages/serve/src/protocols/anthropic-messages.ts`
- `packages/serve/src/protocols/anthropic-messages-input.ts`
- `packages/serve/src/protocols/anthropic-tool-turns.ts`
- `packages/serve/src/protocols/anthropic-messages-formatting.ts`
- `packages/serve/src/protocols/anthropic-messages.test.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/src/streaming/writer-anthropic-messages.ts`
- `packages/transformers/src/chat-template.ts`

## Reference Check

- Anthropic tool definitions use top-level `tools` with `name`, `description`, and `input_schema`; names follow `^[a-zA-Z0-9_-]{1,64}$`.
- Anthropic messages carry `tool_use` blocks in assistant content and `tool_result` blocks in user content. Tool results must immediately follow the assistant tool-use turn, and `tool_result` blocks must precede any user text in the same content array.
- Anthropic `tool_choice` is object-shaped. This tranche supports `auto` and `none`; `any` and named `tool` remain rejected because forced tool use has assistant-prefill semantics that are not implemented here.
- Anthropic streaming tool use has its own content-block and JSON-delta event shape. This tranche rejects `stream: true` when active tools are present.

References:

- https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
- https://platform.claude.com/docs/en/build-with-claude/streaming

## Tensor Lifetime Audit

The changed protocol adapters normalize JSON request and response shapes only.
They allocate no `MxArray` handles, create no transformer caches, and do not
change model execution or media tensor preparation. Streaming tool use remains
rejected before generation starts, so `writer-anthropic-messages.ts` does not
gain a partially implemented tool-use SSE path.

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/http/server.test.ts` passed: `51 pass`, `280 expect()`.
- `bun run --filter '@mlxts/serve' typecheck` passed.
- `bun run lint` passed.
- `bun run check:coverage` passed.
- `bun run validate` passed.

No model hot path, cache implementation, scheduler, or tensor preparation path
changed in this tranche. There are no performance claims beyond preserving the
existing non-tool generation path and rejecting streaming tool use before any
engine invocation.

## Independent Review

Bohr reviewed the Anthropic tool-use mapping before implementation and after
the first pass. The review called out the object-shaped `tool_choice` contract,
immediate and complete `tool_result` ordering, `is_error` handling, rich
tool-result content, and streaming-tool-use SSE gap. Those points are covered
by guardrails and tests in this tranche.

## Guardrails

- Generated tool-call extraction only runs when active tools are present. Tool-looking text remains text when no tools are active or `tool_choice: { "type": "none" }` suppresses tools.
- Assistant `tool_use` history requires the next user message to contain the complete, non-duplicated matching `tool_result` set.
- `tool_result is_error=true` is rejected instead of being silently dropped; the internal chat-template contract has no tool-error field.
- Rich `tool_result` image/document content is rejected for this local endpoint. String and text-block tool results are supported.
- Streaming tool use is rejected until `writer-anthropic-messages.ts` owns Anthropic `tool_use` SSE block starts and JSON input deltas.

## Remaining Risks / Follow-ups

- Anthropic server tools, forced tool choice, token-efficient tool use, interleaved thinking, and streaming tool-use deltas remain separate tranches.
- Optional tool-definition fields beyond `name`, `description`, and `input_schema` remain out of scope for this mapping.
- OpenResponses tool widening is still separate from this Anthropic adapter work.

## Out-of-scope Drift Noticed

None.
