# Lazy Pool Bounded Pressure Shedding

## Summary

The explicit `shed_non_pinned` lazy model-pool policy now sheds active requests
incrementally. It evicts idle non-pinned models first; if active pressure
remains, it aborts the oldest eligible active non-pinned request scope, waits a
bounded time for normal lease release, and retries. Repeated pressure can shed
one additional eligible active scope per retry. The default `reject` policy is
unchanged.

## Files Reviewed

- `packages/serve/src/model-loading/pool.ts`
- `packages/serve/src/model-loading/pool-pressure.ts`
- `packages/serve/src/model-loading/pool-types.ts`

## Tensor Lifetime Audit

This tranche changes serving control flow, abort ordering, and timeout handling.
It does not add tensor-producing operations or new `MxArray` ownership. Model
weights are still not disposed while active leases exist; pressure shedding only
aborts cooperative request scopes and waits for the existing release path.

## Memory / Performance Evidence

The default path remains reject-only and does not cancel active work. The
opt-in pressure path now limits blast radius by cancelling one active non-pinned
lease per relief pass instead of every eligible active lease at once. A
non-cooperative active request cannot leave the blocked load/request waiting
forever; the blocked caller receives `model_pool_pressure_timeout`.

Validated:

- `bun run --filter '@mlxts/serve' typecheck`
- `bun test packages/serve/src/model-loading/pool.test.ts` (`23` tests)
- `bun test packages/serve/src/model-loading/pool.test.ts packages/serve/src/model-loading/sources.test.ts packages/serve/src/observability/metrics.test.ts packages/serve/src/cli.test.ts`
- `bun test packages/serve`
- `bun run validate`
- `bun run regression:qwen-gemma -- --profile quick`

## Independent Review

Confucius independently recommended this as the next Phase 9 tranche before
implementation, identifying bounded victim selection, non-cooperative release
timeouts, and pressure smokes as the right continuation after active pressure
policy landed.

Confucius then reviewed the first diff and found two blockers: unrelated lease
releases could extend the pressure wait indefinitely, and concurrent pressure
callers could each abort a separate active lease before retrying. The final
implementation uses an absolute release deadline plus a serialized relief lane;
waiters that observe another completed relief pass retry before shedding another
active lease. Focused tests cover both cases.

Lovelace performed a second blocker-only review and found no code blocker. The
review confirmed default `reject` behavior remains unchanged and the opt-in
`shed_non_pinned` path sheds one active lease per pass, waits for normal
release, retries, and terminates when no relief remains or the bounded release
timeout fires. Lovelace also flagged a non-blocking stream-startup retry edge;
the final patch closes it with a focused test and repeated relief attempts for
retry stream factories that fail before yielding.

## Out-of-scope Drift Noticed

- The victim policy is oldest eligible request scope, not memory-largest or
  operator-priority-based.
- No real-model `shed_non_pinned` Qwen/Gemma pressure smoke is included in this
  tranche.

## Remaining Risks / Follow-ups

- Add real-model pressure smoke coverage before recommending
  `shed_non_pinned` as the default operator policy.
- Use real pressure traces before widening victim selection beyond oldest-first.
- Add shutdown wake coverage for a pending pressure wait if operator traces show
  disposal latency matters.
