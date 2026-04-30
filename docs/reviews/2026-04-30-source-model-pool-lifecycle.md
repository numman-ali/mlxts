# Runtime Review: Source-backed model pool lifecycle

## Summary

`serveModels()` now supports an explicit lazy source load policy for multi-model
serving. Lazy source entries publish model ids at startup, load the requested
checkpoint on first use, share concurrent first loads for the same model, evict
idle non-pinned models after the configured TTL, and keep pinned models resident
until server shutdown.

The default policy remains eager. Single-model eager CLI serving still uses the
existing `serveModel()` path.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/model-loading/pool.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/source-pool-server.ts`
- `packages/serve/src/model-loading/sources.ts`

## Tensor Lifetime Audit

This tranche does not add tensor-producing primitives or new direct `MxArray`
ownership. It changes when loaded model engines are created and disposed.

Loaded lazy models are retained only after the source load, admission metadata,
and per-model engine construction all succeed. If setup fails after
`loadModelEntry()` returns, the loaded model is disposed before the error
propagates. Idle eviction runs only when the pool state is still current,
`activeCount` is zero, and the entry is not pinned. Streaming requests release
their active lease in the async-iterator `finally` path, including early iterator
close. Pool shutdown clears idle timers and disposes loaded engines/models.

Cold loads across different model ids are serialized through the pool load lane
so the existing model-load memory preflight observes earlier completed loads
instead of racing two first loads against the same active-memory snapshot.

## Memory / Performance Evidence

Focused evidence:

- `bun test packages/serve/src/model-loading/pool.test.ts packages/serve/src/model-loading/sources.test.ts packages/serve/src/cli.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run check:file-lines`
- `bun run --filter '@mlxts/serve' test`
- `bun run regression:qwen-gemma -- --profile quick`
- `bun run validate`

The focused suite passed after the queued-shutdown blocker fix with 33 tests.
The serve package test suite passed after the final blocker fix with 384 tests.
The quick Qwen/Gemma regression passed with 84 transformer-focused tests and
222 serve-focused tests.
Full repo validation passed before commit.

The implementation preserves eager startup as the default and keeps the existing
per-loaded-model generation engine unchanged. Lazy mode moves model load cost
from startup to first request and serializes cold loads to preserve memory
preflight safety.

## Independent Review

Avicenna performed an independent second-opinion review of the uncommitted
tranche. The first pass found two blockers: concurrent cold lazy loads could
bypass the active-memory preflight, and setup failure after `loadModelEntry()`
could leak the loaded model. Both blockers were fixed and covered by regression
tests.

The follow-up pass found one remaining lifecycle blocker: a queued cold load
could start after pool shutdown once the serialized load lane opened. The pool
now re-checks the stopped state after acquiring the load lane and before
calling the source loader, with a regression test covering the queued shutdown
case. The final follow-up review reported no remaining blockers.

## Remaining Risks / Follow-ups

No out-of-scope drift was changed.

HTTP-level lazy streaming with an idle TTL is covered by the lower-level pool
stream lease test rather than a protocol-level SSE test. A future serving
hardening pass can add a full SSE lazy eviction regression if stream lifecycle
bugs appear around the HTTP layer.
