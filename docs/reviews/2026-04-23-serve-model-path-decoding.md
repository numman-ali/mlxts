# Runtime Review: Serve Model Path Decoding

## Summary

Hardened `GET /v1/models/{model}` path parsing so encoded model ids such as
`org%2Fmodel` retrieve the expected served model, while malformed percent-encoded
paths return an OpenAI-shaped 400 instead of escaping as a generic server error.

## Files Reviewed

- `packages/serve/src/protocols/openai-models.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

No tensor, MLX, or native-resource code changed. The update only parses route
metadata and formats JSON errors before any model execution path is reached.

## Memory / Performance Evidence

Validation used the Bun fetch handler with fake engines.

- `bun test packages/serve/src/server.test.ts`
- `bun run check:file-lines`

## Independent Review

This was selected as a narrow serving robustness gap while a separate explorer
audits the next OpenAI-compatible serving tranche. The change keeps model-route
parsing in the protocol adapter rather than adding more inline server logic.

## Remaining Risks / Follow-ups

This only hardens model retrieval path decoding. Broader OpenAI-client behavior
such as CORS/preflight and additional compatibility aliases should be handled as
separate protocol slices with focused tests.
