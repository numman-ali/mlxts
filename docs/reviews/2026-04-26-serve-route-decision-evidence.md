# Serve Route Decision Evidence

## Summary

Serving now emits explicit route-decision events before transformer-backed
generation runs. Benchmarks preserve those decisions in JSON and print a compact
route summary, so concurrency evidence can distinguish true static or continuous
batching from single-request fallback.

This is an observability and correctness tranche. It does not claim new Qwen or
Gemma batching support. Instead, Qwen hybrid-cache requests and Gemma
sliding/global-cache requests now report why they stayed on the single-request
path.

## Files Reviewed

- `packages/serve/src/types.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/src/model-server.test.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`

## Runtime Invariants

- Route decisions are emitted as serving events, not inferred from benchmark
  concurrency after the fact.
- The event separates `single`, `static`, and `continuous` routes and records
  whether the request was eligible for the chosen batching path.
- Fallback reasons stay operator-facing: streaming, max batch size,
  single-request groups, unsupported model type, sliding-window cache, sampled
  generation, and repetition penalty.
- Qwen and Gemma cache fallbacks remain explicit until their cache semantics are
  represented in the scheduler layer.

## Tensor Lifetime Audit

This tranche does not add tensor-producing operations or change model/cache
ownership. It adds route metadata around existing execution paths and benchmark
report extraction from serving events.

## Independent Review

Two read-only sub-agent audits informed this tranche. One audit found that
Qwen/Gemma serving had stability evidence but insufficient route/fallback
evidence, making it too easy to confuse serialized concurrency with batching.
The other audit found that Qwen's remaining gap is likely peak footprint and
long-prefill measurement, not a decode hot-path regression. That points to this
order: make serving evidence unambiguous first, then profile Qwen long-prefill
memory directly against `mlx-lm`.

## Memory / Performance Evidence

- `bun test packages/serve/src/transformers-engine.test.ts`
- `bun test packages/serve/src/cli.test.ts`
- `bun test packages/serve/src/model-server.test.ts packages/serve/scripts/benchmark-serve.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-server.test.ts packages/serve/scripts/benchmark-serve.test.ts`
- `bun run typecheck`
- `bun run lint`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`

## Remaining Risks / Follow-ups

- Benchmark JSON now records route decisions, but it still does not include full
  per-request queue wait, prefill duration, decode duration, or silent-stream
  interval metrics. Those are the next observability slice before high-quality
  long-prefill scheduler claims.
- Continuous batching is still full-KV greedy only. Qwen hybrid cache, Gemma
  layer-pattern/sliding cache, sampling, and streaming need separate scheduler
  work.
