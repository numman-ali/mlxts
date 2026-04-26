# Runtime Review: Qwen/Gemma Regression Guardrails

## Summary

This tranche fixes two Qwen 3.6 loading regressions and adds a package-owned
regression matrix for the current high-risk model families. MLX quantized
embeddings now stay packed instead of becoming dense embedding tables, and
generic `loadCausalLM()` treats top-level Qwen 3.5 / 3.6 configs as text-only
causal LMs while preserving full vision-wrapper loading through an explicit
Qwen conditional loader.

The regression matrix is intentionally split into a cheap focused test path and
an opt-in real-model path. Real-model mode loads cached Qwen 3.6 and Gemma 4
sequentially under the shared runtime lock, resets memory accounting per model,
and can run a strict decode smoke with memory and throughput budgets.

## Files Reviewed

- `examples/qwen3_5-image/index.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/quantized-embedding.ts`
- `packages/quantize/src/index.ts`
- `packages/quantize/src/quantize-module.ts`
- `packages/quantize/src/setup-quantized-module.ts`
- `packages/quantize/src/traversal.ts`
- `packages/quantize/src/types.ts`
- `packages/transformers/package.json`
- `packages/transformers/scripts/regression-model-matrix.ts`
- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/load.ts`
- `packages/transformers/src/families/qwen3_5/weights.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/load.ts`
- `packages/transformers/src/quantize.ts`

## Tensor Lifetime Audit

- `QuantizedEmbedding.forward()` gathers packed weights, scales, and optional
  quantization biases for selected rows only, then frees those selected
  intermediates in `finally`.
- `QuantizedEmbedding.asLinear()` uses MLX `quantizedMatmul` directly for tied
  output projection and does not materialize the dense embedding matrix.
- `quantizeModule()` and `setupQuantizedModule()` replace eligible embeddings
  and linears through module child slots, then dispose the previous dense child.
- Qwen text-only causal loading maps language-model weights into the text tree
  and treats observed vision prefixes (`vision_tower.`, `model.visual.`) as
  ignored for generic `loadCausalLM()`, relying on existing loader ownership for
  skipped tensors.
- The regression matrix clears allocator cache and resets peak memory before
  each real-model load so reported active/peak memory is scoped to that model.

## Memory / Performance Evidence

- `bun run --filter '@mlxts/transformers' regression:models`
  - 66 focused tests passed across quantized embedding, quantize setup, Qwen
    3.5/3.6 weights/model loading, Gemma 4 weights/model loading, and shared
    pretrained loading.
- `bun run packages/transformers/scripts/regression-model-matrix.ts --decode-smoke`
  - The decode smoke invokes `bun run bench:generation:parity`, the paired
    generation benchmark surface. This is the targeted `bench:generation`
    evidence for this tranche.
  - Qwen: `mlx-community/Qwen3.6-27B-4bit`, `model_type=qwen3_5_text`,
    load active `15.134GB`, decode rung `1024x128`, generation `29.359 tok/s`,
    peak `17.184GB`, active slope `0.14 MB/token`, `1.00` evals/token.
  - Gemma 4: `google/gemma-4-E2B-it`, load active `9.295GB`, decode rung
    `1024x128`, generation `81.554 tok/s`, peak `9.893GB`, active slope
    `-0.04 MB/token`, `1.00` evals/token.
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run check:coverage`
  - Coverage thresholds satisfied across the canonical package stack.

## Independent Review

Two read-only sub-agent reviews informed this tranche. One review found that
real-model loads needed the shared runtime lock, isolated peak/cache accounting,
strict decode thresholds, and explicit coverage for the new Qwen conditional
loader. The other review found that the Qwen text-only default and
`QuantizedEmbedding` design were directionally correct, but that multimodal
loader coverage and public quantization wording needed tightening.

Those findings were integrated before this review was written.

## Remaining Risks / Follow-ups

- The decode smoke is a pre-commit guardrail, not the full long-context ladder.
  Qwen/Gemma publishable capability claims still need staged context/output
  benchmarks and paired mlx-lm references where available.
- The explicit Qwen conditional loader has fixture coverage, but a full real
  multimodal image smoke should remain a separate acceptance rung because it
  loads the vision tower and exercises image preprocessing.
- The regression matrix is package-local today. If this becomes the canonical
  local pre-commit command, document the exact cheap and real-model invocations
  in the package README or repo validation docs.
