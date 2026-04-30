# Lazy Pool Active Memory Shedding

## Summary

The source-backed lazy model pool now has an explicit memory-pressure policy.
The default `reject` policy preserves existing behavior: model-load and
request-memory failures reject the blocked request without touching active
requests. `shed_non_pinned` first evicts idle non-pinned models, then aborts
active non-pinned request scopes with `model_pool_memory_pressure`, waits for
normal generation cleanup, and retries the blocked load/request once. Pinned
models are not pressure-shed.

## Files Reviewed

- `packages/serve/src/cli.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli-flag-readers.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/model-loading/pool.ts`
- `packages/serve/src/model-loading/pool-pressure.ts`
- `packages/serve/src/model-loading/pool-types.ts`
- `packages/serve/src/model-loading/source-pool-server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/observability/metrics.ts`
- `packages/serve/src/observability/pool-pressure-metrics.ts`
- `packages/serve/src/types.ts`

## Tensor Lifetime Audit

This tranche changes serving control flow, abort propagation, events, metrics,
and CLI/API policy plumbing. It does not add tensor-producing operations or new
`MxArray` ownership. Active-request shedding aborts the linked request signal
and waits for existing generation cleanup paths to release KV and scheduler
reservations.

## Memory / Performance Evidence

Default behavior remains reject-only, so existing active requests are not
cancelled unless the operator selects `shed_non_pinned`. In the shedding policy,
the pool never disposes active model weights directly; it cancels cooperative
request scopes and waits for lease release before retrying once. Focused tests
cover idle eviction, pinned refusal, active stream shedding for cold-load
pressure, request-time `memory_budget_exceeded` retry, stream-startup error
preservation, CLI policy parsing, and pressure metrics.

Validated:

- `bun run --filter '@mlxts/serve' typecheck`
- `bun test packages/serve/src/model-loading/pool.test.ts packages/serve/src/model-loading/sources.test.ts packages/serve/src/observability/metrics.test.ts packages/serve/src/cli.test.ts`
- `bun test packages/serve`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run validate`
- `bun run regression:qwen-gemma -- --profile quick`

## Independent Review

Confucius reviewed the first pass and caught a real stream-startup regression:
the initial async-generator wrapper deferred startup errors until iteration. The
final implementation starts the inner stream inside `engine.stream()` again,
preserving the existing startup rejection and lease-release contract while
retaining one retry for pre-output memory-budget failures. The review also
flagged unconditional active shedding as a product regression risk, so the
final surface makes shedding operator-explicit through `modelPressurePolicy`.

## Out-of-scope Drift Noticed

- Victim selection is all eligible non-pinned active leases, not smallest-first
  or oldest-first.
- Pressure wait has no timeout for a non-cooperative model that ignores abort.

## Remaining Risks / Follow-ups

- Add a bounded victim-selection policy once real multi-model pressure traces
  show whether oldest, largest estimated memory, or per-model priority works
  best.
- Add a timeout/fallback path for non-cooperative active leases.
- Run a real-model `shed_non_pinned` smoke under Qwen/Gemma memory pressure
  before recommending it as the default operator policy.
