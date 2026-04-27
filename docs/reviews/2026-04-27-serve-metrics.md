# Runtime Review: Serve Metrics Surface

## Summary

`@mlxts/serve` now exposes `GET /metrics` as a Prometheus text-format operator
surface. Metrics are derived from the existing `ServeEvent` stream, so HTTP
routing, generation lifecycle, route decisions, model-lane waits, scheduler
phases, batch starts, prefill chunks, token counts, and MLX memory telemetry
share one observation path.

The route follows the existing auth posture: `/health` stays open, while
`/metrics` is protected when `apiKey` / `--api-key` is configured. Scrapes of
`/metrics` are excluded from HTTP counters to keep Prometheus polling from
dominating local signals.

## Files Reviewed

- `packages/serve/src/serve-metrics.ts`
- `packages/serve/src/serve-metrics-registry.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/server-events.ts`
- `packages/serve/src/server-generation.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/server.test.ts`
- `packages/serve/src/serve-metrics.test.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/README.md`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-optimization-matrix.md`
- `MEMORY.md`

## Tensor Lifetime Audit

This tranche does not add or move MLX tensor-producing operations. The new
collector stores primitive counters, gauges, and histogram buckets from already
emitted event data. It does not touch model forward, cache update, sampling, or
tokenizer decode paths.

The only runtime wiring change is event-sink composition: first-class model
servers create one collector and pass a metrics-recording sink into transformer
engines while passing the same collector to the HTTP router. This keeps engine
events and router events in one scrape surface without adding metrics calls to
hot model loops.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/src/serve-metrics.test.ts packages/serve/src/server.test.ts`
- `bun run typecheck`

The metrics path is host-side event accounting and response rendering, so no
heavy MLX benchmark was required for this slice. The collector uses bounded
labels: model-route paths collapse to `/v1/models/:model`, `/metrics` is
self-excluded, and known served-model lists collapse unknown model labels to
`__unknown__`.

## Independent Review

Dewey reviewed the design before integration. The key guidance was to derive
metrics from `ServeEvent`, avoid request-id/prompt/error-message labels, protect
`/metrics` like `/info`, exclude `/metrics` scrapes from HTTP metrics, normalize
dynamic paths, and make first-class model serving share one collector between
engine events and HTTP router events.

The implementation follows that guidance. It intentionally avoids client-side
quantile summaries in the first pass; Prometheus histograms are the initial
standard surface.

## Remaining Risks / Follow-ups

The current surface is production-useful but not a complete serving dashboard.
Future cache backends should add cache hit/miss/eviction/utilization metrics
when those states exist. Streaming-specific byte/chunk metrics remain in the
benchmark report path today and should become production counters when SSE chunk
events are promoted into the serve event stream.
