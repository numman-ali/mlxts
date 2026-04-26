# Runtime Review: Serve Benchmark Observability

## Summary

This slice strengthens `bench:serve` as the evidence surface for serving-quality
work before deeper Qwen/Gemma scheduler changes. Reports now include per-request
client metrics and benchmark-observed server event timelines, so long-prefill,
streaming, and staggered concurrency runs can be inspected without relying only
on trial averages.

No model hot path or serving production event contract changed. The benchmark
records existing serve events with local observation timestamps.

## Files Reviewed

- `AGENTS.md`
- `MEMORY.md`
- `PLAN.md`
- `docs/product-surfaces.md`
- `packages/serve/README.md`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/benchmark-serve-completions.ts`
- `packages/serve/scripts/benchmark-serve-completions.test.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/transformers/README.md`

## Tensor Lifetime Audit

No tensor-producing production code changed. The benchmark additions are
host-side metrics over HTTP responses and already-emitted serve events.

## Evidence

- `bun test packages/serve/scripts/benchmark-serve-completions.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/serve/src/cli.test.ts`
- `bun test packages/serve/scripts/benchmark-serve-completions.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run lint`

## Independent Review

Two explorer sub-agents independently recommended improving serving telemetry
before changing Qwen/Gemma scheduler semantics. A worker sub-agent implemented a
bounded docs-alignment slice, which was reviewed by the lead agent; one overclaim
about persisted per-model settings was corrected to future work before
validation.

## Remaining Risks

- Server-side timings are benchmark-observed event timings, not a new production
phase event contract.
- Queue wait and scheduler-internal phase telemetry still need a dedicated
production event design before Qwen/Gemma batching claims.
- Real-model proof should use `bun run regression:qwen-gemma -- --profile real`
or substantial profiles when cached checkpoints and machine time are available.
