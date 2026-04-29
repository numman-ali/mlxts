# Continuous Scheduler Memory Budget

## Summary

Continuous serving now reserves estimated request memory through the same
model-level admission controller that already reserves prompt, completion, and
aggregate scheduled tokens. The byte estimate stays in `@mlxts/serve`, where
model geometry, configured `gpuMemoryUtilization`, and MLX allocator telemetry
already live. `@mlxts/transformers` only carries generic scheduler pressure
fields and the `scheduled_memory_budget` deferral reason.

## Files Reviewed

- `packages/serve/src/admission/continuous-budget.ts`
- `packages/serve/src/admission/continuous-memory.ts`
- `packages/serve/src/engine/continuous.ts`
- `packages/serve/src/observability/scheduler-metrics.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-admission.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-events.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`

## Runtime Sensitivity Notes

`continuous-budget.ts` is the shared admission controller for all continuous
scheduler keys on a loaded serve model. Adding memory pressure there prevents
separate scheduler keys from independently admitting rows that each passed
single-request preflight but collectively exceed the configured model-level
headroom.

`continuous-memory.ts` snapshots MLX headroom once when the transformer serving
engine creates the shared controller. This makes the scheduler budget
deterministic for the engine lifetime and keeps per-admission checks host-side.
It is not a serve-wide allocator and does not replace per-request active-memory
preflight.

## Tensor Lifetime Audit

This tranche changes host-side admission metadata only. It does not allocate,
retain, or dispose `MxArray` values. Existing scheduler cleanup still disposes
admission reservations through `cleanupScheduledRequest()`, which now releases
the request's estimated memory reservation alongside its token reservation.

## Memory / Performance Evidence

- `bun run --filter '@mlxts/transformers' typecheck`: passed.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun test packages/serve/src/admission/continuous-budget.test.ts packages/serve/src/cli.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/serve/src/observability/metrics.test.ts`: passed, `34` tests.
- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`: passed, `29` tests.
- `bun test packages/serve/scripts/regression-serve-matrix.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/src/admission/continuous-budget.test.ts`: passed, `22` tests after adding regression memory-pressure evidence.
- `bun test packages/serve`: passed, `307` tests.
- `bun run validate`: passed.
- `bun run regression:qwen-gemma -- --profile real`: passed after the final
  scheduler-budget code path. Decode smoke kept Qwen at `28.982` generation
  tokens/s with `1.00` evals/token and Gemma at `81.579` generation tokens/s
  with `1.00` evals/token. Serving regression kept
  Qwen and Gemma continuous routes green, including Qwen concurrency `8`, Gemma
  concurrency `8`, protocol health for chat/responses/Anthropic, and the
  long+short fairness smokes. The Qwen `32768x128+128x32` fairness rung passed
  with `peak_memory=19.277` GB and `routes=continuous:eligible=2`.
- `bench:generation`: not rerun. This tranche does not change model forward
  math, sampling, cache tensor update, or per-token decode execution.
- `bench:generation:parity`: covered through the real Qwen/Gemma regression
  decode smoke. The real serving regression remains the relevant endpoint proof
  because this change affects scheduler admission and reporting.

## Independent Review

Singer reviewed the design before final validation and agreed with extending the
existing scheduler admission contract while keeping byte estimation in
`@mlxts/serve`. The review called out the main limitation: the creation-time
headroom snapshot is deterministic scheduler backpressure, not a live global
memory budget across prompt-prefix cache growth, fallback/static paths, or
multiple loaded model engines.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

The next memory-management layer is a serve-wide model pool and allocator:
model-load estimates, idle eviction, pinned models, and active-request abort
policy when one loaded model must shed KV pressure. This tranche deliberately
does not implement that pool.
