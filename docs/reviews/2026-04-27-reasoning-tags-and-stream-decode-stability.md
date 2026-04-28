# Reasoning Tags And Stream Decode Stability

## Summary

This tranche hardens local serving and agent compatibility for Pi-style Qwen usage:
known reasoning wrappers are split out of visible assistant text, `thinking off`
preserves explicit `preserve_thinking` chat-template replay hints from clients,
and streaming decode no longer commits an unstable trailing UTF-8 replacement
character before the next token can complete it.
Reasoning tag normalization now lives in the zero-dependency `@mlxts/protocols`
package so serving and agent clients share the same Qwen, Anthropic-style, and
Gemma thought-channel parsing behavior.

## Files Reviewed

- packages/serve/src/protocols/anthropic-messages-formatting.ts
- packages/serve/src/protocols/anthropic-messages.ts
- packages/serve/src/protocols/openai-chat-completion-streaming.ts
- packages/serve/src/protocols/openai-chat-completions.ts
- packages/serve/src/protocols/openai-responses.ts
- packages/serve/src/protocols/openai-responses-formatting.ts
- packages/serve/src/protocols/openai-usage.ts
- packages/serve/src/protocols/reasoning-tags.ts
- packages/agent/src/reasoning-tags.ts
- packages/protocols/src/index.ts
- packages/protocols/src/index.test.ts

## Tensor Lifetime Audit

The protocol changes are string parsing and request-normalization only. They add
no tensor-producing operations, no `MxArray` ownership changes, and no FFI calls.
The stream decode stability change in `transformers-engine-streaming.ts` only
changes emitted string deltas around tokenizer output; it does not allocate or
retain model tensors.

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/protocols/openai-responses.test.ts`
- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/transformers-engine.test.ts`
- `bun test packages/agent/src/chat-model.test.ts`
- `bun test packages/protocols/src/index.test.ts packages/agent/src/chat-model.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`

No throughput benchmark was run for this protocol-only slice because the changed
serving files do not alter model forward math, cache mutation, or scheduling.
The Pi/Qwen regression test now covers the cache-sensitive shape where Pi sends
`enable_thinking: false` with `preserve_thinking: true`; an exact-only Qwen-like
snapshot is reused on the next turn instead of recomputing the whole prefix.
OpenAI-compatible usage now reports `cached_tokens` as cache reads only, keeping
cache writes in the explicit `cache_write_tokens` field so Pi's `R`/`W` footer
does not double-count first-time writes as reads.

## Independent Review

Sub-agent Halley inspected Pi's local docs and installed source. It confirmed
Pi maps `qwen-chat-template` thinking levels to `chat_template_kwargs` booleans,
streams visible text from `delta.content`, and treats `reasoning_content` as the
correct hidden-thinking channel. Halley found no Pi runtime source for
`<antThinking>`, making the mlxts-side compatibility parser the correct seam.

Sub-agent Bacon later reviewed the cache regression and confirmed that Qwen
hybrid snapshots can only fork at exact stored offsets. The concrete regression
was in the protocol adapter: Pi explicitly sends `preserve_thinking: true` for
Qwen chat-template replay, and forcing it to `false` made the next rendered
multi-turn prompt diverge before the reusable cache boundary.

## Remaining Risks / Follow-ups

Tool parameter streaming is still not fully end-to-end: `@mlxts/agent` consumes
OpenAI-style streamed tool-call argument deltas, but `@mlxts/serve` buffers
model-native `<tool_call>...</tool_call>` envelopes until the JSON is complete.
Full Pi/opencode parity needs an incremental tool-call envelope stream or a
model prompt format where the tool name and arguments can be emitted separately.
