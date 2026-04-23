# Runtime Review: Serve OpenAI Responses Adapter

## Summary

Added a deliberately narrow `POST /v1/responses` route over the existing
protocol-neutral generation engine. The adapter supports text-only `input`,
optional `instructions`, `max_output_tokens`, model-native sampling omission,
`seed`, `metadata`, and non-persistent local `store: false`. It returns an
OpenAI-style `response` object with `output`, `output_text`, usage, and reasoning
items when the engine supplies reasoning content.

Unsupported Responses features are rejected explicitly instead of silently
pretending they work: stateful continuation, conversations, background jobs,
streaming, tools, files/images, prompt templates, truncation, includes, reasoning
controls, and non-text output formats.

## Files Reviewed

- `packages/serve/src/index.ts`
- `packages/serve/src/protocols/openai-responses.ts`
- `packages/serve/src/protocols/openai-responses-formatting.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

No tensor, MLX, cache, or native-resource code changed. Responses requests
normalize into the same `NormalizedGenerationRequest` path as completions and
chat completions, then call the existing engine `generate()` contract.

## Memory / Performance Evidence

Validation used fake generation engines only; no large live Qwen/Gemma servers
were started for this protocol-only tranche.

- `bun test packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/server.test.ts`
- `bun test packages/serve/src`
- `bun run lint`
- `bun run typecheck`
- `bun run check:assertions`
- `bun run check:file-lines`

## Independent Review

Dalton audited `@mlxts/serve` for the next safe serving-quality improvement and
identified `/v1/responses` as the highest-leverage protocol gap that can be
closed without touching model execution or large-model validation. The adapter
shape was checked against OpenAI's official Responses create reference.

## Remaining Risks / Follow-ups

This is intentionally not full Responses coverage. Streaming Responses,
stateful continuation, tool calls, multimodal inputs, prompt templates, and
background jobs still need separate protocol and engine tranches. The next live
serving validation should still focus on controlled Qwen/Gemma smoke tests with
the existing generation logs, not on this protocol shim.
