# Runtime Review: Qwen/Gemma Continuous Streaming

## Summary

Qwen 3.6 text and Gemma 3/4 layer-pattern greedy streaming requests now route
through the same cache-generic continuous scheduler as buffered greedy
generation. The serving route stays narrow: sampled/model-native-default
requests still fall back to the single-request path until batched sampling is
implemented.

This matters for real multi-agent local serving because streamed completions and
chat turns no longer need to serialize purely because the request is streaming,
as long as the model family already has proven batch-cache semantics.

## Files Reviewed

- `packages/serve/src/transformers-engine-routing.ts`

## Tensor Lifetime Audit

The production change is route-only. It removes the streaming-specific
ineligibility guard for models that already pass the Qwen model-owned batch
cache or Gemma layer-pattern batch-cache checks.

No new MLX tensor-producing operations were added. Streaming still flows through
`createContinuousTransformersGeneration()` and `ContinuousBatchTokenScheduler`,
which own active caches, prefilling caches, current-token tensors, request abort
signals, and iterator cleanup. The new tests exercise early iterator closure for
Qwen and Gemma streams so scheduler rows are cancelled instead of silently
decoding to the requested maximum.

## Memory / Performance Evidence

Focused checks:

- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts`
- `bun run packages/serve/scripts/regression-serve-matrix.ts --real-models --qwen-model mlx-community/Qwen3.6-27B-4bit --gemma4-model google/gemma-4-E2B-it --report-dir .tmp/qwen-gemma-regression-continuous-streaming --request-timeout-ms 3600000`

Focused route and budget tests passed. The scheduler-backed early-close tests
for Qwen and Gemma streams passed and observed `cancelled` scheduler phases.

The real Qwen/Gemma endpoint matrix passed. Key evidence:

- Qwen streaming `1024x128@1`: `routes=continuous:eligible=1`,
  `continuous_admissions=1`, `continuous_admission_rows=1`,
  `continuous_scheduler_phases=4`, `max_generation_batch=1`,
  `post_ttft_completion_tps=24.979`, `max_stream_chunk_gap_ms=82.8`,
  `peak_memory=17.514 GB`, `active_delta=0.000 GB`.
- Qwen non-streaming simultaneous `128x32@2`: `routes=continuous:eligible=2`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `peak_memory=16.244 GB`, `active_delta=0.000 GB`.
- Qwen non-streaming staggered `128x32@2`: `routes=continuous:eligible=2`,
  `continuous_admissions=2`, `continuous_admission_rows=3`,
  `continuous_scheduler_phases=9`, `max_generation_batch=2`,
  `peak_memory=15.904 GB`, `active_delta=0.000 GB`.
- Gemma 4 streaming `1024x128@1`: `routes=continuous:eligible=1`,
  `continuous_admissions=1`, `continuous_admission_rows=1`,
  `continuous_scheduler_phases=4`, `max_generation_batch=1`,
  `post_ttft_completion_tps=63.045`, `max_stream_chunk_gap_ms=32.8`,
  `peak_memory=10.404 GB`, `active_delta=0.000 GB`.
- Gemma 4 non-streaming `128x32@2`: `routes=continuous:eligible=2`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `peak_memory=9.671 GB`, `active_delta=0.000 GB`.

## Independent Review

Carson independently audited the proposed route widening before this artifact.
The review concluded that the route is mechanically safe for the narrow greedy,
text-only Qwen 3.6 and Gemma 3/4 streaming case because the streaming adapter is
already cache-generic and scheduler-backed, but it should only ship with real
endpoint SSE cadence/cancellation evidence.

Carson also highlighted the main lifecycle risk: a stream consumer may close
after one emitted token while the scheduler is about to sample the next token.
The added early-close tests cover that class by asserting scheduler cancellation
for Qwen and Gemma streams.

## Remaining Risks / Follow-ups

This does not add sampled batched decode. Model-native default sampling still
routes to the single-request path.

The real matrix proves single-request streamed Qwen/Gemma scheduler routing and
batched buffered routing. A future tranche should add explicit concurrent
streaming `@2` real-model rungs once the regression matrix can afford the extra
runtime, so we can assert streamed batch size `2` through SSE telemetry as well
as through unit tests.

Stop sequences are still enforced by the SSE adapter rather than as
scheduler-native string stops. Existing stream writers call `iterator.return()`
on stop-filter exits, and the new early-close tests cover iterator cleanup, but
textual stop behavior should remain part of future protocol-level streaming
regressions.
