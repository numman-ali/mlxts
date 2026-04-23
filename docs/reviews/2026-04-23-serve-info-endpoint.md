# Runtime Review: Serve Info Endpoint

## Summary

Added a lightweight `GET /info` operator endpoint for local serving. It reports
the served model ids, enabled wire endpoints, configured request limits, and
whether the active engine exposes streaming or batch generation. `/health`
remains open, while `/info` uses the same bearer API-key check as `/v1/*` routes
when `--api-key` is configured.

The endpoint intentionally avoids local filesystem paths, cache directories,
access tokens, live queue internals, and claims of continuous batching.

## Files Reviewed

- `packages/serve/src/index.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

No tensor, MLX, cache, model forward, or native handle code changed. The endpoint
formats metadata already held by the server and does not call the generation
engine.

## Memory / Performance Evidence

Validation used fake generation engines only; no large live Qwen/Gemma servers
were started for this metadata-only tranche.

- `bun test packages/serve/src/server.test.ts packages/serve/src/model-server.test.ts`
- `bun run typecheck`

## Independent Review

Hilbert audited the proposed introspection step and recommended a small `GET
/info` outside `/v1`, protected by the same API key as OpenAI routes. The shape
was cross-checked against TGI's `/info` convention and oMLX's lightweight status
endpoint, while keeping mlxts-specific claims deliberately narrower.

## Remaining Risks / Follow-ups

`batch_generation: true` only means the current engine has a `generateBatch`
function; it does not claim every request shape uses static batching or
continuous batching. Live queue depth, memory counters, and Prometheus-style
metrics should be separate endpoints once those counters are first-class.
