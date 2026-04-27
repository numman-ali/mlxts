# Runtime Review: Continuous Scheduler Split Budget

## Summary

Continuous scheduler admission now preserves separate model-level pressure for
prompt reservations, completion reservations, and the aggregate total-token
envelope. This closes the gap where multiple continuous scheduler keys could be
bounded by row count and split fields while accidentally weakening the existing
`maxTotalTokens * maxBatchSize` guard.

The same split pressure is now visible in scheduler events, CLI logs, Prometheus
metrics, benchmark reports, and serving regression evidence. Deferrals carry a
bounded reason: prompt budget, completion budget, or aggregate scheduled-token
budget.

## Files Reviewed

- `packages/serve/src/continuous-scheduler-budget.ts`
- `packages/serve/src/continuous-scheduler-budget.test.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/src/serve-scheduler-metrics.ts`
- `packages/serve/src/serve-metrics.test.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-admission.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-events.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-telemetry.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `docs/serving-runtime-strategy.md`
- `PLAN.md`
- `MEMORY.md`

## Tensor Lifetime Audit

This change is host-side scheduler admission and telemetry only. It does not add
new MLX tensor-producing operations, cache tensor mutation, sampler kernels, or
model forward-path tensor expressions.

Reservations are still released by the scheduler cleanup path through disposable
admission reservations. The split budget tracks host-side counters only:
scheduled prompt tokens, scheduled completion tokens, and scheduled total
tokens. Release listeners remain release-driven, which avoids the hot-loop shape
where a partially free budget repeatedly wakes a row that still cannot fit.

## Memory / Performance Evidence

Focused validation passed after the split and aggregate-cap fix:

- `bun test packages/serve/src/continuous-scheduler-budget.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/serve-metrics.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run typecheck`
- `bun run lint`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`
- `bun run validate`
- `bun run regression:qwen-gemma -- --profile quick`

The focused scheduler tests cover prompt/completion deferrals, total-token
deferrals, oversize prompt/completion/total rejection, and release-only wakeups.
The serve engine tests cover Qwen and Gemma continuous routing, streaming,
sampled/model-default requests, and cancellation through the same scheduler
event surface.

Real endpoint smoke passed for both cached Qwen and Gemma checkpoints:

- Qwen 3.6 27B 4bit:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --mixed-rungs 1024x32+128x16 --trials 1 --greedy --ignore-eos --stream --report-json .tmp/qwen36-split-budget-mixed-smoke.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --request-stagger-ms 100 --request-timeout-ms 3600000 --no-warmup`
  completed with `7.682` completion tok/s end-to-end, `19.497` post-TTFT
  tok/s, `17.514 GB` peak memory, `2` continuous admissions, `3` continuous
  admission rows, and max generation batch size `2`.
- Gemma 4 E2B:
  `bun run bench:serve --model google/gemma-4-E2B-it --model-id gemma-local --mixed-rungs 1024x32+128x16 --trials 1 --greedy --ignore-eos --stream --report-json .tmp/gemma4-split-budget-mixed-smoke.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --request-stagger-ms 100 --request-timeout-ms 3600000 --no-warmup`
  completed with `38.708` completion tok/s end-to-end, `37.516` post-TTFT
  tok/s, `10.404 GB` peak memory, `2` continuous admissions, `3` continuous
  admission rows, and max generation batch size `2`.

`bench:generation` and `bench:generation:parity` were not rerun for this slice.
The changed production code is host-side serving admission, event typing,
logging, benchmark reporting, and metrics; it does not alter model math,
sampling kernels, cache tensor layout, or the per-token transformer forward
path. Endpoint evidence is the relevant performance surface for this change.

## Independent Review

Singer reviewed the design before implementation and recommended keeping one
shared model-level controller across scheduler keys while splitting prompt and
completion pressure, preserving release-driven wakeups, and propagating bounded
deferred reasons through events, metrics, and benchmark reports.

Ramanujan reviewed the dirty diff independently and caught the key bug in the
first split implementation: the old aggregate `maxTotalTokens` envelope had been
weakened by reporting and enforcing `promptCap + completionCap`. The final
implementation carries an explicit aggregate scheduled-total cap when
`maxTotalTokens` is configured and tests that shape directly.

## Remaining Risks / Follow-ups

This is still a scheduled-token reservation budget, not byte-accurate cache
reservation. Future cache backends should expose per-row byte estimates so the
same admission seam can reserve actual cache memory.

The next serving-quality passes should focus on stronger fairness policy under
`@4` and `@8` mixed load, plus prefix/paged/quantized KV backends behind typed
strategy seams. The split budget is the guardrail that keeps those experiments
from over-admitting while they mature.
