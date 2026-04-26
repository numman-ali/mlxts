# Runtime Review: Cache-Generic Continuous Batching

## Summary

The continuous token scheduler now creates the most specific batch cache a model
can consume instead of hard-coding `BatchKVCache`. Static and continuous
generation share the same cache factory: model-owned caches first, Gemma
layer-pattern caches second, plain full-KV cache last.

Gemma 3/4 greedy non-streaming serving can now use the continuous scheduler with
`LayerPatternBatchKVCache`. Gemma streaming remains single-route fallback, and
Qwen remains static-only in serving even though `Qwen3_5TextBatchCache.extend()`
now exists as a prerequisite for future continuous proof.

## Files Reviewed

- `packages/serve/src/transformers-engine-routing.ts`
- `packages/transformers/src/families/qwen3_5/batch-cache.ts`
- `packages/transformers/src/infrastructure/generation/batch-cache-factory.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`

## Tensor Lifetime Audit

`createBatchCacheForModel()` centralizes cache creation without taking ownership
of model state. The scheduler continues to own and dispose active caches,
prefilling caches, and current-token tensors in the existing `finally` and
failure paths.

`Qwen3_5TextBatchCache.extend()` builds all next linear conv/recurrent arrays
before mutating the target cache. It rejects wrong cache classes, mismatched
layer types, non-exhausted linear padding, asymmetric linear state
initialization, and mismatched non-batch dimensions/dtypes. Full-attention state
extension remains delegated to `BatchKVCache.extend()`. Prepared linear arrays
are disposed on failure before target state is replaced.

The continuous scheduler helpers now operate on `TransformerBatchCache`, and the
new prefilling-cache disposal helper only moves existing ownership cleanup out
of the scheduler file; it does not retain or duplicate native handles.

`bun run check:assertions` reports no production type assertions,
`bun run check:tensor-lifetimes` reports no suspicious nested tensor-producing
calls, and `bun run check:file-lines` reports all active production files are
under the 500-line cap.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/families/qwen3_5/model.test.ts packages/transformers/src/families/gemma3/model.test.ts packages/transformers/src/families/gemma4/model.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run check:assertions`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16`
- `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-cache-generic-continuous`

Focused transformer tests passed: 26 pass, 0 fail. Focused serve tests passed:
36 pass, 0 fail. The real Qwen/Gemma wrapper passed 70 transformer checks and
134 serve checks before endpoint rungs.

Direct Qwen generation evidence:

- `bench:generation`: `prompt_tps=254.079`, `generation_tps=29.179`,
  `peak_memory=17.184 GB`, `active_slope_mb_per_token=0.13`,
  `evals_per_token=1.00`.
- `bench:generation:parity --skip-mlx-lm-reference`: `prompt_tps=245.609`,
  `generation_tps=29.363`, `peak_memory=17.184 GB`,
  `active_slope_mb_per_token=0.14`, `evals_per_token=1.00`.

Real regression decode smoke:

- Qwen parity smoke: `generation_tps=29.102`, `peak_memory=17.184 GB`,
  `active_slope_mb_per_token=0.14`, `evals_per_token=1.00`.
- Gemma 4 parity smoke: `generation_tps=82.180`, `peak_memory=9.893 GB`,
  `active_slope_mb_per_token=-0.04`, `evals_per_token=1.00`.

Real endpoint evidence:

- Qwen streaming `1024x128@1` stayed `single:unsupported_model_type=1`,
  with zero static/continuous counters, `28.321 tok/s` post-TTFT, and
  `17.184 GB` peak memory.
- Qwen non-streaming `128x32@2` stayed static-only:
  `static:eligible=2`, `static_batches=1`, `static_batch_rows=2`,
  `continuous_admissions=0`, `continuous_scheduler_phases=0`,
  `max_generation_batch=2`, `25.051 tok/s`, `16.244 GB` peak memory.
- Gemma 4 streaming `1024x128@1` stayed single-route fallback:
  `single:streaming=1`, zero static/continuous counters, `81.381 tok/s`
  post-TTFT, and `9.892 GB` peak memory.
- Gemma 4 non-streaming `128x32@2` now uses continuous scheduling:
  `continuous:eligible=2`, `continuous_admissions=1`,
  `continuous_admission_rows=2`, `continuous_scheduler_phases=7`,
  `max_continuous_batch=2`, `max_generation_batch=2`,
  `static_batches=0`, `60.044 tok/s`, `9.671 GB` peak memory.

## Independent Review

Turing reviewed the scheduler/cache seam and recommended making the continuous
scheduler generic over `TransformerBatchCache`, routing Gemma non-streaming
behind tests, and keeping Qwen continuous disabled until `Qwen3_5TextBatchCache`
extension was implemented and proven.

Chandrasekhar reviewed Qwen `extend()` semantics independently and recommended
delegating full-attention extension to `BatchKVCache`, concatenating linear
conv/recurrent state only on the batch axis, rejecting asymmetric state, and
keeping Qwen continuous serving disabled until delayed-admission endpoint
evidence exists. The final implementation follows that recommendation.

## Remaining Risks / Follow-ups

Qwen continuous serving is still deliberately disabled. The hybrid cache can
extend now, but the route should not widen until staggered Qwen scheduler tests
with the real model and endpoint-level evidence prove delayed row admission,
chunked prefill, filtering/cancellation, and memory stability.

Gemma streaming remains a single-route fallback. Routing it into continuous
would need streaming-specific endpoint evidence, not just the non-streaming
batch proof in this tranche.
