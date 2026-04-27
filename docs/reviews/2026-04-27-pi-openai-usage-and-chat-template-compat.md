# Runtime Review: Pi OpenAI Usage and Chat Template Compatibility

## Summary

This change keeps the local Pi provider path aligned with the full served model id and fills the OpenAI-compatible usage details Pi parses for token/cache footer metrics. It also covers the earlier chat-template normalization that lets Qwen replay OpenAI wire-format tool-call arguments through Jinja templates expecting mappings.

## Files Reviewed

- `packages/serve/src/types.ts`
- `packages/serve/src/protocols/openai-usage.ts`
- `packages/serve/src/protocols/openai-completions.ts`
- `packages/serve/src/protocols/openai-chat-completion-streaming.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/transformers/src/chat-template.ts`

## Tensor Lifetime Audit

No tensor-producing code, native handles, model-cache state, or MLX evaluation paths were changed. The serve changes are protocol JSON formatting only. The chat-template change normalizes host-side message objects before Jinja rendering and does not create or retain `MxArray` values.

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/openai-completions.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server.test.ts packages/serve/src/server-streaming.test.ts`
- `bun test packages/transformers/src/chat-template.test.ts packages/serve/src/model-context.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run typecheck`
- Live Qwen smoke against `mlx-community/Qwen3.6-27B-4bit` returned a streamed OpenAI chat usage chunk with `prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 }`.
- Live Pi TUI smoke used `--model mlx-community/Qwen3.6-27B-4bit`, executed `read package.json`, answered `mlxts`, and showed the full model id plus token/context footer metrics.

## Independent Review

Sub-agent Godel the 2nd independently inspected Pi's installed footer and provider parsing paths. The finding was that Pi v0.70.2 renders cumulative input/output/cache/cost/context/model footer metrics, parses OpenAI `prompt_tokens_details` for cache reads/writes, and does not render TTFT/TPS directly in the installed footer source.

## Remaining Risks / Follow-ups

`@mlxts/serve` currently reports zero cache read/write tokens because prefix-cache reuse is not implemented yet. TTFT/TPS are available through serve stream logs, `/metrics`, and benchmark reports; surfacing them inside Pi's footer would require a Pi extension status line or an upstream Pi UI change.
