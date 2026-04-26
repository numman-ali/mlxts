# Runtime Review: Sampled Continuous Batching

## Summary

The continuous scheduler now supports sampled and model-native-default decode
for cache-batch-capable models. It still performs one batched model forward per
decode step, but each active request owns its own `SamplerState`, so RNG state
and repetition history stay row-local while cache filtering/admission remain
scheduler-owned.

Serving now routes eligible sampled/default Qwen 3.6 text and Gemma 3/4
requests through continuous batching for buffered and streaming completions.
Static batch generation remains greedy-only.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/transformers/src/infrastructure/sampling/runtime.ts`
- `packages/transformers/src/infrastructure/sampling/index.test.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/README.md`

## Tensor Lifetime Audit

Each scheduled request now constructs a `SamplerState` at enqueue time. Normal
finish, queued cancellation, prefilling cancellation, active-row cancellation,
and scheduler failure all call `#cleanup()`, which removes abort listeners and
disposes the sampler state.

The sampled decode helper keeps the greedy fast path batched. For sampled or
repetition-penalty paths it slices row logits, samples with the request's own
`SamplerState`, evaluates each token before local row handles are released, and
concatenates retained token rows into the next `[batch, 1]` token tensor. The
combined token tensor is freed by the existing scheduler lifecycle.

The scheduler failure path now rejects prefilling rows before disposing their
caches. This closes a hang/leak class where chunked prefill failures could clear
`#prefilling` before `#failAll()` saw those requests.

Gemma 4 exposed an MLX `arange` failure when building a static top-k mask after
a large-vocabulary sampled graph. The top-k masked-position vector is now a
one-time host-side int32 constant, avoiding that graph-shape failure while
keeping filtering on device.

## Memory / Performance Evidence

Focused checks:

- `bun test packages/transformers/src/infrastructure/sampling/index.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run packages/serve/scripts/regression-serve-matrix.ts --real-models --qwen-model mlx-community/Qwen3.6-27B-4bit --gemma4-model google/gemma-4-E2B-it --report-dir .tmp/qwen-gemma-regression-sampled-continuous-v3 --request-timeout-ms 3600000`
- `bench:generation` / `bench:generation:parity`: not rerun for this tranche because
  the changed behavior is scheduler route/sampling orchestration rather than a
  model-forward kernel optimization. The endpoint regression matrix above is
  the capability and performance evidence for this serving change.

The real Qwen/Gemma endpoint matrix passed, including model-default sampled
buffered and streamed rungs:

- Qwen 3.6 model-default buffered `128x16@2`: `routes=continuous:eligible`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `completion_tps=17.348`, `peak_memory=16.245 GB`, `active_delta=0.000 GB`.
- Qwen 3.6 model-default streamed `128x16@2`: `routes=continuous:eligible`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `post_ttft_completion_tps=24.731`, `stream_chunks=32`,
  `peak_memory=16.245 GB`.
- Gemma 4 model-default buffered `128x16@2`: `routes=continuous:eligible`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `completion_tps=53.400`, `peak_memory=9.672 GB`, `active_delta=0.000 GB`.
- Gemma 4 model-default streamed `128x16@2`: `routes=continuous:eligible`,
  `continuous_admissions=1`, `continuous_admission_rows=2`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `post_ttft_completion_tps=36.960`, `stream_chunks=7`,
  `stream_bytes=2459`, `peak_memory=9.672 GB`.

## Independent Review

Socrates audited the sampled batching design before implementation and
recommended batched forward with per-row sampler state rather than simply
removing the route guard.

Poincare reviewed the uncommitted diff and found the prefilling failure cleanup
bug, the missing runtime review artifact, over-broad streaming evidence wording,
and missing seeded sampling coverage. The implementation now includes those
fixes or narrowed budgets.

## Remaining Risks / Follow-ups

Sampling is correct and scheduler-backed, but row-wise sampling is not the final
performance ceiling. A future tranche can evaluate whole-batch categorical
sampling once per-row seed semantics and repetition-history behavior can be
preserved without making results batch-composition-dependent.

Gemma sampled streaming can produce rows with no visible text chunks before
`done` because sampled tokens may decode to skipped or empty visible text. The
regression therefore requires aggregate SSE evidence, server request completion,
route decisions, and scheduler counters rather than per-request visible TTFT for
that specific sampled short-output rung.
