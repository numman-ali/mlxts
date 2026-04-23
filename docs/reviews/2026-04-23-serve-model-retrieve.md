# Runtime Review: Serve Model Retrieve

## Summary

Added OpenAI-compatible single-model retrieval at `GET /v1/models/{model}` alongside the existing model list endpoint. Unknown model ids now return the same shaped `model_not_found` error used by generation routing.

## Files Reviewed

- `packages/serve/src/index.ts`
- `packages/serve/src/protocols/openai-models.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

No tensor, MLX, or native-resource code changed. The route only reads served model metadata and formats JSON responses.

## Memory / Performance Evidence

Validation used the Bun fetch handler with fake engines.

- `bun test packages/serve/src/server.test.ts`
- `bun run typecheck`
- `bun run check:file-lines`

## Independent Review

This was selected from the serving ergonomics scan as a small OpenAI-client compatibility gap. The behavior mirrors the existing list-model response shape and the router's `model_not_found` error semantics.

## Remaining Risks / Follow-ups

Model ids containing slashes should be URL-encoded by clients when using the retrieve route. The route decodes the trailing path segment after `/v1/models/`, so encoded ids continue to work.
