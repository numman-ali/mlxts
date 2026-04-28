# Serve Src Folder Restructure Review

## Summary

Restructured `packages/serve/src/` into role-named folders without changing
serving behavior. The root now keeps only CLI, barrel, error, and type files;
HTTP, streaming, engine, admission, runtime, observability, model-loading, and
media concerns live under their matching folders.

## Files Reviewed

- `packages/serve/scripts/benchmark-serve-options.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/src/admission/batching.ts`
- `packages/serve/src/admission/concurrency.ts`
- `packages/serve/src/admission/continuous-budget.ts`
- `packages/serve/src/admission/request-limits.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/engine/batch.ts`
- `packages/serve/src/engine/content.ts`
- `packages/serve/src/engine/continuous.ts`
- `packages/serve/src/engine/execution-lane.ts`
- `packages/serve/src/engine/generation.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/routing.ts`
- `packages/serve/src/engine/shared.ts`
- `packages/serve/src/engine/static.ts`
- `packages/serve/src/engine/streaming.ts`
- `packages/serve/src/http/abort.ts`
- `packages/serve/src/http/events.ts`
- `packages/serve/src/http/json.ts`
- `packages/serve/src/http/route-generation.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/http/routes-anthropic.ts`
- `packages/serve/src/http/routes-responses.ts`
- `packages/serve/src/http/server.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/media/image.ts`
- `packages/serve/src/model-loading/router.ts`
- `packages/serve/src/model-loading/server-options.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/observability/metrics-registry.ts`
- `packages/serve/src/observability/metrics.ts`
- `packages/serve/src/observability/scheduler-metrics.ts`
- `packages/serve/src/observability/stream-metrics.ts`
- `packages/serve/src/runtime/memory.ts`
- `packages/serve/src/runtime/model-context.ts`
- `packages/serve/src/runtime/strategy.ts`
- `packages/serve/src/streaming/heartbeat.ts`
- `packages/serve/src/streaming/lifecycle.ts`
- `packages/serve/src/streaming/observability.ts`
- `packages/serve/src/streaming/runtime.ts`
- `packages/serve/src/streaming/stop-filter.ts`
- `packages/serve/src/streaming/writer-anthropic-messages.ts`
- `packages/serve/src/streaming/writer-openai-responses.ts`
- `packages/serve/src/streaming/writer-openai.ts`

No route extraction, OpenAI writer split, or shared writer-base abstraction was
introduced in this tranche; those are behavior-preserving candidates for the
later stream-writer extraction tranche.

## Tensor Lifetime Audit

No tensor-producing expressions, native handle ownership, cache mutation, or
model execution logic changed. Moved runtime-sensitive files retain the same
local tensor lifetime visibility as before; changed lines are import/export path
updates and benchmark test path strings.

## Memory / Performance Evidence

No performance claim is made. The restructure does not change scheduling,
streaming cadence, cache shape, admission limits, or model-lane behavior.

## Validation

- `bun run typecheck`
- `bun test packages/serve/src packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts` passed: 275 tests.

## Independent Review

No sub-agent review was run in this tranche because this session did not include
explicit delegation authorization. The review was constrained to mechanical
path-diff inspection, serve-focused tests, typecheck, and the required full
repo gate before commit.

## Out-of-scope Drift Noticed

- `routes-openai.ts` remains folded into `http/server.ts` because extraction is
  not a pure file move.
- `streaming/writer-base.ts` and separate OpenAI completions/chat writers remain
  for the dedicated stream-writer scaffolding tranche.

## Remaining Risks / Follow-ups

Historical docs and memory entries still reference the old file paths. They are
not live imports, but final continuity and memory cleanup should record the new
layout once all tranches land.
