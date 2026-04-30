# Anthropic Streaming Tool Use

## Summary

`/v1/messages` now accepts `stream: true` when active client tools are present.
The Anthropic SSE writer parses generated tool-call envelopes and emits
Anthropic-shaped `tool_use` content blocks with `input_json_delta` argument
deltas, then finishes the message with `stop_reason: "tool_use"`.

This is a protocol formatting change only. `@mlxts/serve` still does not
execute tools; agents and external clients own tool execution and follow-up
`tool_result` turns.

## Files Reviewed

- `packages/serve/src/protocols/anthropic-messages.ts`
- `packages/serve/src/streaming/writer-anthropic-messages.ts`
- `packages/serve/src/streaming/writer-anthropic-messages.test.ts`
- `packages/serve/src/protocols/anthropic-messages.test.ts`
- `packages/serve/src/http/server.test.ts`

## Reference Check

Official Anthropic streaming docs define the Messages stream as
`message_start`, one or more content-block lifecycles, `message_delta`, and
`message_stop`. Tool-use streaming opens a `tool_use` content block, streams the
tool input through `content_block_delta` events whose delta type is
`input_json_delta`, closes the block, and reports `stop_reason: "tool_use"` in
the message delta.

Source checked on 2026-04-30:

- https://platform.claude.com/docs/en/build-with-claude/streaming

## Runtime Review

The implementation keeps the existing serve boundary:

- Generated tool-call parsing reuses the conservative serving parser already
  used by OpenAI Chat and OpenResponses streams.
- Tool-looking text stays visible when no tools are active.
- Text and reasoning blocks keep the existing Anthropic lifecycle.
- Stop-sequence filtering applies only to visible text, not structured tool
  arguments.
- The terminal stream summary still reports the engine finish reason and usage;
  only the Anthropic wire `stop_reason` becomes `tool_use` when a tool block was
  emitted.

## Tensor Lifetime Audit

No tensor-producing code, native handles, cache state, model execution, or MLX
evaluation paths changed. The diff only changes request normalization and SSE
JSON framing over already-generated text chunks.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/src/streaming/writer-anthropic-messages.test.ts packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/http/server.test.ts`
- Result: `56 pass`, `313 expect()` calls.
- `bun run validate`
- `bun run regression:qwen-gemma -- --profile quick`
  - Transformer-focused regressions: `84 pass`.
  - Serve-focused regressions: `220 pass`.

No live model benchmark was required for this tranche because model execution
and scheduler/cache code are untouched.

## Independent Review

Kant independently recommended Anthropic Messages streaming tool-use as the next
bounded Phase 9 slice after the OpenResponses streaming tool tranche. The review
called out the main risk as wire-shape correctness and specifically named the
`content_block_start` / JSON-delta / `content_block_stop` lifecycle as the
contract to preserve.

Bohr recommended Phase 8 training-proof hardening as the next broader roadmap
slice. That does not conflict with this protocol tranche; it is the next
candidate after this commit.

## Guardrails

- Server-side tool execution is still out of scope.
- `tool_choice: "any"` and forced named-tool choice remain rejected until those
  semantics are implemented explicitly.
- Built-in/server tools, token-efficient tool use, fine-grained eager input
  streaming, and rich `tool_result` media remain separate tranches.

## Remaining Risks / Follow-ups

- The stream writer emits one complete argument JSON delta per generated tool
  envelope. That is sufficient for current generated-envelope parsing, but not
  Anthropic's finer-grained `eager_input_streaming` beta behavior.
- A live Anthropic-compatible client smoke through a real served model should be
  added when the next Pi/agent validation session is run.

## Out-of-scope Drift Noticed

Phase 8 training-proof reporting still needs a machine-checkable verifier before
the fine-tuning proof can be treated as a stronger product gate.
