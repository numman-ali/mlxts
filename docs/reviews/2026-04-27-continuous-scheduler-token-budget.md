# Runtime Review: Continuous Scheduler Token Budget

## Summary

Continuous batching now has an explicit scheduled-token budget seam. The
transformer scheduler can defer waiting rows when a shared admission controller
cannot reserve their prompt-plus-generation tokens, and serving wires one
model-level controller across all continuous scheduler keys for a loaded model.

The first serving budget is derived from existing operator limits:
`maxTotalTokens * maxBatchSize`. It is intentionally not a new public flag. The
goal is to prevent multiple scheduler keys from silently exceeding the model's
configured concurrent token envelope while keeping row-count batching,
chunked-prefill fairness, and model-native sampling behavior intact.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-admission.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-events.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-lifecycle.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-telemetry.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-wakeup.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/serve/src/continuous-scheduler-budget.ts`
- `packages/serve/src/continuous-scheduler-budget.test.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/serve-metrics.ts`
- `packages/serve/src/serve-scheduler-metrics.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/README.md`
- `docs/serving-runtime-strategy.md`
- `docs/inference-optimizations.md`
- `MEMORY.md`

## Tensor Lifetime Audit

This change does not introduce new MLX tensor-producing operations. The model
forward, sampling, cache update, and token tensor paths stay on the existing
continuous scheduler implementation.

The new reservation object is host-side only. Reservations are released from
the scheduler cleanup path, which already runs for normal finish, cancellation,
and failure. Prefilling cache disposal remains explicit on abort/failure paths,
and the abort lifecycle helper preserves the existing cache disposal behavior.

`continuous-batch.ts` remains at the file-size gate by moving host-side
admission, abort lifecycle, and wakeup mechanics into small helper modules
rather than hiding tensor ownership inside larger nested expressions.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/src/continuous-scheduler-budget.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/serve-metrics.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run lint`
- `bun run typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`
- `bun run validate`
- `bench:generation` and `bench:generation:parity` were not run for this
  slice because no model math, cache tensor layout, sampling kernel, or
  per-token decode tensor operation changed. The runtime-sensitive change is
  host-side scheduler admission and telemetry; endpoint/model throughput
  evidence should come from the real Qwen/Gemma `regression:serve` matrix after
  this guardrail lands.

The new scheduler test proves that a second row is deferred when its reservation
would exceed the scheduled-token budget, then proceeds after the first row
releases its reservation. Focused serving tests verify continuous routes still
work for Qwen/Gemma cache shapes, streaming, sampled defaults, cancellation,
and benchmark/report evidence.

## Independent Review

Russell reviewed local reference scheduler designs in `mlx-lm`, `vllm-mlx`,
`omlx`, and `rapid-mlx`. The recommendation was to keep MLX execution
serialized, make scheduler budgets explicit, and defer paged KV, SSD cache,
speculative decode, and TurboQuant-style backends until budget/fairness evidence
is solid.

Banach reviewed the local serving scheduler code and caught the key production
shape: one loaded model can own multiple continuous scheduler instances keyed by
batch/sampling options, so a budget that lives only inside one scheduler can
undercount aggregate admitted work. The implementation follows that guidance by
wiring one shared model-level admission controller into every scheduler created
by `createContinuousTransformersGeneration()`.

Kant reviewed the uncommitted diff and caught a partial-capacity wakeup risk:
release listeners must not wake immediately merely because some budget is free,
because the waiting row may still not fit and could hot-loop. The serving budget
now wakes listeners only on an actual reservation release, with a regression test
covering that shape.

## Remaining Risks / Follow-ups

This is a scheduled-token reservation budget, not full byte-accurate memory
reservation. Per-request memory preflight still uses model config geometry, and
future work should add aggregate byte reservations once cache backends expose
more precise per-row memory.

The next scheduler passes should split prefill and completion budgets, add
explicit prefill-chunk budget events, and run higher-concurrency real Qwen/Gemma
benchmarks before claiming production-grade fairness under `@4` or `@8` loads.
