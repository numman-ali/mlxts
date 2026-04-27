# Runtime Review: Serve Stream Observability

## Summary

`@mlxts/serve` now emits server-side stream telemetry through the same
`ServeEvent` path as request, generation, scheduler, and batch observability.
The SSE writers observe output-bearing frames and terminal stream state, then
`/metrics` reports bounded stream counters and histograms for terminal result,
server-side TTFT, SSE/output frame counts, byte counts, and stream duration.

This keeps benchmark/client TTFT separate from server-side TTFT: the production
metric measures generation start to first output-bearing SSE frame inside the
server, while the benchmark harness remains responsible for client-observed
network timing.

## Files Reviewed

- `packages/serve/src/server-stream-runtime.ts`
- `packages/serve/src/server-stream-observability.ts`
- `packages/serve/src/server-stream-lifecycle.ts`
- `packages/serve/src/serve-stream-metrics.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/serve-metrics.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/server.test.ts`
- `packages/serve/src/server-streaming.test.ts`
- `packages/serve/src/serve-metrics.test.ts`
- `packages/serve/README.md`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-optimization-matrix.md`
- `MEMORY.md`

## Tensor Lifetime Audit

This tranche does not add or move MLX tensor-producing operations. The new code
observes already serialized SSE frames and records primitive counters,
histogram values, byte lengths, and timestamps.

The stream observer is attached at the protocol writer boundary rather than in
model generation. That avoids per-model tensor ownership changes and keeps stop
filtering, reasoning/tool parsing, terminal `[DONE]`, cancellation, and writer
errors in one observation seam.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/src/serve-metrics.test.ts packages/serve/src/server-streaming.test.ts packages/serve/src/server.test.ts`
- `bun run typecheck`

No heavy MLX benchmark was required for this slice because the change is
host-side stream accounting. The implementation avoids high-cardinality labels:
request id, prompt, raw errors, and raw paths are not metric labels. Per-output
chunk events are suppressed from CLI output unless verbose mode is enabled.

## Independent Review

Anscombe reviewed the local serving code and recommended deriving stream
telemetry from the SSE writers rather than porting benchmark parsing logic.
Bohr reviewed reference serving stacks and highlighted first-token latency,
stream lifecycle, bounded labels, cancellation, and queue/prefill/decode phase
separation as the useful observability pattern.

The implementation follows that guidance for the streaming slice while leaving
broader cache-hit, active scheduler, and per-request debug-state metrics for the
cache/scheduler tranches where those states become first-class.

## Remaining Risks / Follow-ups

`generation_stream_chunk` events are output-frame events, not token events.
Throughput claims should continue to use usage/token metrics and benchmark
reports rather than SSE chunk counts.

Heartbeats are intentionally not counted as output chunks. Future dashboard
work can add a separate heartbeat metric if operator evidence shows it is
useful.
