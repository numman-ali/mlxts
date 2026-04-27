# Runtime Review: Anthropic Messages Serving

## Summary

Added bounded Anthropic Messages compatibility to `@mlxts/serve` without forking model execution. `/v1/messages` now parses text-only Anthropic requests into `NormalizedGenerationRequest`, returns Anthropic message objects, emits Anthropic SSE events, preserves reasoning as thinking blocks, and uses Anthropic-shaped errors for that route. Unsupported images/tools are rejected explicitly.

## Files Reviewed

- `packages/serve/src/errors.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/protocols/anthropic-messages.ts`
- `packages/serve/src/protocols/anthropic-messages-formatting.ts`
- `packages/serve/src/server-anthropic-messages.ts`
- `packages/serve/src/server-anthropic-messages-streaming.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/server.ts`
- `packages/serve/scripts/benchmark-serve-completions.ts`
- `packages/serve/scripts/benchmark-serve-options.ts`

## Tensor Lifetime Audit

This change does not create, hold, dispose, or reshape MLX tensors. It stays above the model/runtime layer: JSON request parsing, normalized request construction, SSE event formatting, route dispatch, and benchmark parsing. The existing generation engine remains responsible for tensor ownership and streaming iterator cleanup.

## Memory / Performance Evidence

Focused protocol and route tests passed:

```bash
bun test packages/serve/src/protocols/anthropic-messages.test.ts \
  packages/serve/src/server.test.ts \
  packages/serve/scripts/benchmark-serve-completions.test.ts \
  packages/serve/scripts/benchmark-serve-options.test.ts
```

Result: 52 pass, 0 fail. This covers request normalization, response formatting, Anthropic SSE reasoning/text separation, stop-sequence early stream closure, `/info` endpoint exposure, stream-not-supported errors, malformed JSON, and benchmark parsing for buffered and streaming Anthropic events.

No Qwen/Gemma endpoint benchmark was required for this slice because the model hot path and scheduler were not changed.

## Independent Review

Galileo the 2nd reviewed the serving protocol surface before implementation. Key guidance integrated: implement Anthropic as its own adapter rather than wrapping OpenAI chat formatting, add `/v1/messages` to generation timeout handling and `/info`, use Anthropic SSE event names without OpenAI `[DONE]`, parse benchmark usage from `message_delta`, and add a runtime review artifact.

## Remaining Risks / Follow-ups

This is intentionally text-only. Anthropic image blocks, tool use/tool result blocks, richer thinking signatures, token-count endpoints, and broader Messages compatibility still need dedicated implementation and acceptance tests before they should be advertised as supported.
