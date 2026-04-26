# Runtime Review: Gemma Static Batch Mask Elision

## Summary

Gemma 3/4 layer-pattern static batching now reuses the single-token decode
fast path when there is no effective left padding to mask. For an unpadded
single-token decode step, the causal mask and left-padding mask are both
semantically unnecessary, so the attention path returns `null` instead of
building a left-padded mask tensor.

This fixes a real Gemma 4 endpoint regression where the `128x32@2` static
batch rung crashed inside MLX `arange` after prefill. The benchmark harness now
also records `streamDecodeInterval` and real-model serving budgets assert
streaming TTFT/chunk-gap responsiveness, not just endpoint liveness.

## Files Reviewed

- `packages/transformers/src/infrastructure/cache/layer-pattern-batch.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/families/gemma3/attention.ts`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/serve/scripts/benchmark-serve-options.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`

## Tensor Lifetime Audit

The Gemma fix removes a tensor allocation from the hot path. Layer-pattern
batch caches can now expose effective left-padding values as host numbers; the
existing `leftPaddingTensorForLayer()` still materializes an `MxArray` only for
mask shapes that actually need it.

For `queryLength === 1` with all effective left padding equal to zero, attention
returns a `null` mask. No `MxArray` is created for that case, and no disposable
handle is hidden inside a nested expression. Mixed-length or padded batch decode
still takes the existing explicit left-padded mask path.

The benchmark and regression changes are host-side option/report/budget logic
only. They do not create native arrays or change model cache ownership.

## Memory / Performance Evidence

Direct real-model repro after the fix:

```text
loaded gemma4 262144 35
ok [ 32, 32 ]
```

Focused checks:

- `bun test packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/transformers/src/infrastructure/masks.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/gemma3/model.test.ts packages/transformers/src/families/gemma4/model.test.ts packages/serve/src/transformers-engine.test.ts`
- Result: 78 tests passed, 0 failed.

Real Qwen/Gemma regression:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-stream-budget`
- The transformer decode smoke invokes `bun run bench:generation:parity`
  through the package regression matrix for Qwen 3.6 and Gemma 4, with
  `--skip-mlx-lm-reference` because this tranche is guarding local serving and
  decode stability rather than publishing paired mlx-lm speed claims.
- Qwen direct decode: `28.907 tok/s`, peak `17.184 GB`, active slope `0.14 MB/token`, `1.00` evals/token.
- Gemma 4 direct decode: `81.473 tok/s`, peak `9.893 GB`, active slope `-0.04 MB/token`, `1.00` evals/token.
- Qwen streaming endpoint: `14.613 tok/s` end-to-end, `28.651 tok/s` post-TTFT, TTFT `4326.4 ms`, max stream chunk gap `144.1 ms`, `126` chunks, route `single:unsupported_model_type`.
- Gemma 4 streaming endpoint: `75.105 tok/s` end-to-end, `80.878 tok/s` post-TTFT, TTFT `133.9 ms`, max stream chunk gap `25.0 ms`, `6` chunks, route `single:sliding_window_cache`.
- Gemma 4 static batch endpoint: `64.685 tok/s`, `static_batches=1`, `static_batch_rows=2`, `max_generation_batch=2`, route `static:eligible`.

The generated serve reports include `streamDecodeInterval=1`, making the
responsiveness configuration part of the evidence instead of an implicit server
default.

## Independent Review

Worker sub-agent Mill independently reproduced the same Gemma 4 static batch
failure and identified the same smallest safe fix: skip creating the
left-padded mask when `queryLength === 1` and all effective left padding values
are zero. That review also recommended a future optimization to share Gemma
batch masks per forward for mixed-length padded batches.

Explorer sub-agent Sartre reviewed the serving regression strategy and
recommended making stream responsiveness an asserted harness property before
running heavier ladders. This tranche adds the missing benchmark option/report
field and TTFT/chunk-gap budget checks.

## Remaining Risks / Follow-ups

This does not widen Gemma into continuous batching. Gemma streaming and
sampled/model-native-default requests still stay on the single route until the
scheduler can represent layer-pattern cache extension and filtering directly.

Mixed-length padded Gemma static batches still build explicit masks per layer.
That path is correct, but it may be worth sharing equivalent masks within a
forward pass if benchmarks show repeated mask construction as a real cost.

Qwen still routes through the single fallback because hybrid recurrent/cache
semantics are not represented by the static or full-KV continuous schedulers
yet. The next serving-quality tranche should improve scheduler fairness and
then continue Qwen/Gemma-specific cache semantics deliberately.
