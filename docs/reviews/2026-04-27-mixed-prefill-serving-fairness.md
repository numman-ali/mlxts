# Runtime Review: Mixed Long-Prefill Serving Fairness

## Summary

This tranche fixes and guards mixed long-prefill plus short-arrival serving for
Qwen 3.6 and Gemma 4. The continuous scheduler now starts newly waiting rows
even while another row is still chunk-prefilling, and `@mlxts/serve` passes a
`512` token prefill chunk to favor fairness for heterogeneous arrivals.

The regression matrix now supports request-shape-specific budgets so mixed
rungs can assert the short request's TTFT, stream cadence, and scheduler queue
time without pretending the long request should have a low TTFT.

Gemma 4 chunked prefill also exposed repeated `mlx_arange` fragility while
building mask coordinate vectors. Mask range vectors are deterministic
host-side indices, so `masks.ts` now builds them from `Int32Array` rather than
native `arange`.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/infrastructure/masks.test.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-safety.md`
- `PLAN.md`
- `MEMORY.md`

## Tensor Lifetime Audit

`continuous-batch.ts` changes scheduler control flow only: it can start a newly
waiting row while another row is prefilling, then advances one prefill chunk.
It does not introduce new tensor-producing expressions or new cache ownership.
The new unit test verifies the short row receives a token between long-prefill
chunks and that the long prompt is not full-prefilled in one forward pass.

`masks.ts` replaces MLX-native `arange` calls with host-created `Int32Array`
index vectors. The returned arrays are still local `using` values inside mask
construction and are freed by existing explicit resource management. The mask
math is unchanged: query/key coordinate tensors still feed the same causal,
padding, and sliding-window comparisons.

`transformers-engine-shared.ts` passes `prefillStepSize: 512` through existing
generation option surfaces and memory estimation. This changes chunk sizing but
does not alter cache ownership.

## Memory / Performance Evidence

Focused tests passed:

- `bun test packages/transformers/src/infrastructure/masks.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `bun test packages/serve/scripts/regression-serve-matrix.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/model-context.test.ts`

Direct generation hot-path probes passed after the mask range change:

- `bun run bench:generation --model google/gemma-4-E2B-it --prompt-tokens 512 --generation-tokens 16 --trials 1 --prefill-step-size 512 --memory-sample-interval 16`
  reported `7508.125` prompt tok/s, `86.834` generation tok/s,
  `9.719 GB` peak memory, `0.10 MB/token` active slope, and `1.00`
  evals/token.
- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 512 --generation-tokens 16 --trials 1 --prefill-step-size 512 --memory-sample-interval 16 --skip-mlx-lm-reference`
  reported `6772.942` prompt tok/s, `91.700` generation tok/s,
  `9.719 GB` peak memory, `0.10 MB/token` active slope, and `1.00`
  evals/token. The external mlx-lm reference was intentionally skipped for
  this small hot-path sanity probe; endpoint evidence below is the main proof
  for the mixed serving behavior.

Crash reproduction before the mask fix:

- `bun run bench:serve --model google/gemma-4-E2B-it --model-id gemma-local --rungs 5000x1@1 --trials 1 --greedy --ignore-eos --stream --report-json .tmp/gemma4-single-5000x1-prefill-512.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --request-timeout-ms 3600000 --no-warmup`
- Failed with `MxError: arange failed: [arange] Cannot compute length` at `packages/transformers/src/infrastructure/masks.ts`.

Gemma 4 after the mask fix:

- Single long-prefill probe `5000x1@1`: completed in `764.7ms`, TTFT `760.2ms`,
  peak memory `9.837 GB`, route `continuous:eligible`.
- Mixed `5000x128+128x32`: `47.493` completion tok/s, `38.031`
  post-TTFT tok/s, `601.2ms` mean TTFT, `161.2ms` max stream gap, `9.843 GB`
  peak memory, `10` scheduler phases, `3` continuous admission rows, max
  generation batch size `2`.
- Gemma short request: client TTFT `249.4ms`, server scheduler queue `106.4ms`,
  server stream TTFT `182.5ms`, max silent event gap `156.4ms`.

Qwen 3.6 mixed long/short after the scheduler and prefill changes:

- Before the scheduler fix, short request server queue was `126312.1ms` on the
  `32768x128+128x32` mixed probe.
- After the final fix, mixed `32768x128+128x32`: `0.972` completion tok/s,
  `10.279` post-TTFT tok/s, `4978.6ms` max stream gap, `19.345 GB` peak memory,
  active memory delta `0.014 GB`, `10` scheduler phases, `2` continuous
  admission rows.
- Qwen short request: client TTFT `8681.2ms`, server scheduler queue
  `2604.9ms`, server stream TTFT `4663.3ms`, max silent event gap `4933.1ms`.

Substantial Qwen/Gemma regression passed:

- `bun run regression:qwen-gemma -- --profile substantial --report-dir .tmp/qwen-gemma-regression-mixed-fairness-rerun`
- Qwen long output `1024x1024@1`: `20.654` completion tok/s, `23.013`
  post-TTFT tok/s, `129.5ms` max stream gap, `16.509 GB` peak memory.
- Qwen long context `32768x128@1`: `19.012` post-TTFT tok/s, `152.4ms`
  max stream gap, `19.345 GB` peak memory.
- Qwen mixed `32768x128+128x32`: `9.824` post-TTFT tok/s, `5590.0ms`
  max stream gap, `19.345 GB` peak memory, flat active memory delta, `10`
  scheduler phases, `2` continuous admission rows. The short request client
  TTFT was `7991.1ms`; server scheduler queue was `3115.6ms`; server stream
  TTFT was `5577.6ms`; max silent event gap was `5584.6ms`.
- Gemma mixed `5000x128+128x32`: `49.654` completion tok/s, `38.834`
  post-TTFT tok/s, `156.1ms` max stream gap, `9.843 GB` peak memory, flat
  active memory delta, `10` scheduler phases, `3` continuous admission rows.
  The short request client TTFT was `241.0ms`; server scheduler queue was
  `103.7ms`; server stream TTFT was `180.4ms`; max silent event gap was
  `156.2ms`.
- Qwen 32k retrieval smoke passed early, middle, and late needle placements
  with exact marker matches. Prefill was `183.0`-`186.0` tok/s, decode was
  `22.8`-`23.0` tok/s, peak memory was `23.245 GB`, and active decode slope
  was `0.00 MB/token`.

The Qwen max generation batch size is `1` for this mixed fairness shape because
the short request is admitted and served independently while the long request is
still prefilling. That is the desired fairness behavior for this rung; the
separate `@4` and `@8` simultaneous rungs continue to guard multi-row
generation batching.

## Independent Review

Schrodinger reviewed the budget design and recommended request-shape-specific
assertions correlated through client request id. That avoided the aggregate
TTFT trap where the long request either fails a legitimate budget or forces the
budget to become too loose to prove the short request.

Pasteur reviewed the Gemma 4 crash path and identified `createLeftPaddedAttentionMask`
as the likely fix seam. The investigation confirmed the reported mask lengths
were positive and valid in isolation, and that the durable fix should remove
raw native `arange` from repeated mask coordinate construction rather than
disabling Gemma continuous streaming.

## Remaining Risks / Follow-ups

This tranche proves greedy streaming mixed fairness for the current Qwen/Gemma
capability rungs. It does not yet implement a full production fairness policy
with configurable decode-vs-prefill quotas, nor does it prove sampled mixed
streams at high concurrency.

The serving default `512` prefill chunk is intentionally fairness-biased. Future
operator strategy should expose a throughput/fairness knob once scheduler
policy is first-class, but not before the budgets can prove the chosen mode.
