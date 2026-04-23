# Runtime Review: Serve Generation Error Events

## Summary

Added a structured `generation_error` event so failures after request normalization keep the model id, request id, protocol, error code, message, and duration. The CLI now logs these events by default, making multi-agent and multi-model failures easier to correlate than route-level request errors alone.

## Files Reviewed

- `packages/serve/src/cli.ts`
- `packages/serve/src/server-events.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/types.ts`

## Tensor Lifetime Audit

No tensor-producing or native-resource code changed. The implementation only emits metadata around existing generation failures and preserves the existing error propagation path.

## Memory / Performance Evidence

Validation used fake engines that fail after normalization; no live model serving was needed.

- `bun test packages/serve/src/server.test.ts packages/serve/src/cli.test.ts`
- `bun run typecheck`
- `bun run check:file-lines`

## Independent Review

Hume independently identified the missing generation-level error event as the highest-value serving ergonomics gap for debugging parallel agents and multiple models.

## Remaining Risks / Follow-ups

Streaming cancellation remains modeled as `generation_complete` with `finishReason: "cancelled"` plus request-level `client_cancelled`, not as `generation_error`. That keeps intentional client disconnects distinct from model/runtime failures.
