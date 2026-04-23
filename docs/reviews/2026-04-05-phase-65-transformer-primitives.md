# Runtime Review: Phase 6.5 Transformer Primitives

## Summary

This review covers the runtime-sensitive production changes for Phase 6.5:
modern transformer primitives in the canonical `@mlxts/core` and `@mlxts/nn`
packages. The implementation adds fused RMSNorm and RoPE bindings, reusable
`RMSNorm`, `RoPE`, `GroupedQueryAttention`, and `swiglu` surfaces, the missing
`repeat` / `tile` / `sort` / `topk` ops, exact float16 and bfloat16
safetensors I/O, and low-level MLX quantize/dequantize bindings.

The design stays explicit on purpose. There is no early `transformers` package,
no hidden trainer-style framework, and no model-level abstraction jump ahead of
Phase 7. The new logic lives at the primitive layer where later LLaMA-style
architectures can compose it directly.

## Files Reviewed

- `packages/core/src/array-ffi-data.ts`
- `packages/core/src/fast.ts`
- `packages/core/src/ffi/index.ts`
- `packages/core/src/ffi/lib.ts`
- `packages/core/src/ffi/optional.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
- `packages/core/src/io.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/reduction.ts`
- `packages/core/src/ops/shape.ts`
- `packages/core/src/quantization.ts`
- `packages/nn/src/activations.ts`
- `packages/nn/src/grouped-query-attention.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/rms-norm.ts`
- `packages/nn/src/rope.ts`

## Tensor Lifetime Audit

- New tensor-producing primitives were added to `scripts/runtime-sensitive-ops.ts`
  so the nested-call lifetime gate continues to cover the expanded surface.
- `GroupedQueryAttention` keeps projection, reshape, RoPE, repetition, masking,
  softmax, and residual preparation as explicit locals with `using`, rather
  than nesting disposable intermediates inside larger expressions.
- Exact half-precision safetensors I/O uses explicit contiguous copies and raw
  byte reads from evaluated arrays, rather than hiding dtype widening inside
  convenience conversions.
- Quantize/dequantize stay as low-level core primitives and do not introduce
  hidden module walkers or quantized layer state in this phase.

## Memory / Performance Evidence

Current evidence:

- `cd packages/core && bun test` passes with new fast-op, quantization, and
  safetensors coverage.
- `cd packages/nn && bun test` passes with new RMSNorm, RoPE, GQA, SwiGLU, and
  llama-style integration coverage.
- `cd examples/nanogpt && bun test` passes after the primitive-surface changes,
  including the operator and acceptance-mode regression tests.
- `bun run validate` passes on the final Phase 6.5 tree.

Phase 6.5 is a primitive-surface milestone, so the evidence emphasis here is
correctness, ownership visibility, and package validation rather than long-run
throughput benchmarking.

## Independent Review

Implementation is being done by Codex. Independent review is still required
before this milestone is considered fully closed.

## Remaining Risks / Follow-ups

- `topk` currently follows MLX's native value ordering. If later serving or
  sampling surfaces need a stronger ordering contract, that should be made
  explicit at the higher-level API rather than silently changed here.
- Exact half-precision preservation now exists at the safetensors boundary, but
  the general JS convenience readers (`toTypedArray`, `toList`, `item`) still
  widen half dtypes through float32 for readability. That is intentional for
  now and should remain explicit.
- Phase 7 still needs to define the first real architecture consumer of these
  primitives. This phase intentionally stops at reusable building blocks.
