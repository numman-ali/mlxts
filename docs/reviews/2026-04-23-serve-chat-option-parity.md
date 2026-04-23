# Runtime Review: Serve Chat Option Parity

## Summary

Extended the OpenAI-compatible chat completions adapter so common request fields are either honored or rejected explicitly. This covers `max_completion_tokens`, `seed`, and `user` propagation, plus clear rejection of unsupported non-default controls such as `n > 1`, penalties, logprobs, non-empty `logit_bias`, non-text `response_format`, and `parallel_tool_calls: false`.

## Files Reviewed

- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-chat-messages.ts`

## Tensor Lifetime Audit

No tensor-producing or native-resource code changed. The edits operate entirely on JSON request normalization and metadata shaping before generation reaches an engine.

## Memory / Performance Evidence

This is a protocol-boundary change, so validation used fake request/engine tests rather than live model serving.

- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server.test.ts`
- `bun run check:file-lines`
- `bun run typecheck`

## Independent Review

Feynman independently reviewed the chat/completions protocol adapters and recommended the same scope: carry `seed` and `user`, support `max_completion_tokens`, and explicitly reject unsupported non-default chat semantics instead of silently ignoring them.

## Remaining Risks / Follow-ups

The server still does not implement logprobs, response-format constrained decoding, or single-tool-call enforcement. Those controls now fail loudly rather than silently degrading, and can be implemented behind explicit tests when the runtime capability exists.
