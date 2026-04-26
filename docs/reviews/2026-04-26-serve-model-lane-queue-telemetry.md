# Runtime Review: Serve Model-Lane Queue Telemetry

## Summary

Added a serve-layer model-lane wait event so endpoint benchmarks and CLI logs can distinguish waiting for the local MLX model lane from actual generation work. The benchmark JSON now records model-lane wait timing per generation id, which gives Qwen/Gemma fallback runs a clearer latency breakdown without changing model math, cache semantics, or scheduler behavior.

## Files Reviewed

- `packages/serve/src/model-execution-lane.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/scripts/benchmark-serve.ts`

## Tensor Lifetime Audit

No new tensor-producing operations were added. The changed production code only reads model-lane counters, emits structured events, and wraps existing acquire/release boundaries. The existing `finally { release(); }` ownership pattern remains in place for non-streaming and streaming generation, and no `MxArray` handles are hidden inside the telemetry helpers.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/serve/src/model-execution-lane.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run lint`
- `bun run check:runtime-review`
- `bun run check:tensor-lifetimes`
- `bun run regression:qwen-gemma -- --profile quick`

This is intentionally an observability change. It adds a pair of small event object allocations per single/streaming generation request that enters the model lane, with no extra MLX graph work and no changes to token generation cadence.

## Independent Review

Sub-agent Huygens reviewed the intended queue/phase telemetry seam independently before final validation. Earlier reference review by Ohm recommended queue and phase timestamps as the next industry-standard serving evidence layer, while Lagrange recommended preserving Qwen/Gemma fallback honesty before deeper scheduler work.

## Remaining Risks / Follow-ups

Continuous scheduler internals still need their own scheduler-level queue/step events; this slice only covers the shared model execution lane used by single-route and streaming requests. Future work should add request phase summaries for scheduler waiting, chunked prefill fairness, and prefix/paged cache events once those runtime strategies land.
