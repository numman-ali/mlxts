# Runtime Review: Serve Chat Protocol Compatibility

## Summary

Hardened the OpenAI-compatible chat completions adapter for common client request shapes and streaming usage parity. This review covers message normalization for developer/text-part content and chat SSE chunk formatting when `stream_options.include_usage` is requested.

## Files Reviewed

- `packages/serve/src/protocols/openai-chat-completion-streaming.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`

## Tensor Lifetime Audit

No tensor-producing code changed. The edits are limited to JSON protocol normalization and response formatting before requests reach the model engine, so no MLX array ownership, disposal, stream synchronization, or native handle lifetime changes were introduced.

## Memory / Performance Evidence

Validation used fake protocol/streaming tests only; no live model or heavy MLX run was needed for this compatibility slice.

- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts`
- `bun test packages/serve/src/server-streaming.test.ts`
- `bun run typecheck`

## Independent Review

Bernoulli performed the independent serving-quality scan before implementation and identified the two high-leverage gaps addressed here: chat SSE `usage: null` parity for usage streaming, and support for modern OpenAI message shapes such as `developer` role and text content parts.

## Remaining Risks / Follow-ups

Image content parts are intentionally rejected with a clear error until the serving multimodal path is wired end to end. Broader OpenAI chat fields such as penalties, `n`, logprobs, and response-format controls should get the same explicit support-or-reject treatment in a follow-up protocol-hardening slice.
