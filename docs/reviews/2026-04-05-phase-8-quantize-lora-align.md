# Runtime Review: Phase 8 Quantize, LoRA, and Alignment Foundations

## Summary

This review covers the runtime-sensitive production changes that Phase 8 and
the first Phase 9a-adjacent quantization work add on top of the completed
Phase 7 loading surface. The core shift is from "dense-only checkpoint loading"
to a stack that can prepare quantized module placeholders before assignment,
rewrite dense pretrained snapshots into quantized shard sets, load GGUF weights
through MLX's native GGUF importer, wrap dense or quantized linear layers with
LoRA adapters, and feed alignment recipes from reusable dataset and collation
contracts instead of ad hoc fixture code.

The implementation keeps the package boundaries deliberate. Low-level MLX
quantization primitives stay in `@mlxts/core`, quantized and LoRA-aware layer
forms stay in `@mlxts/nn`, checkpoint-aware quantization setup plugs into the
existing transformers loader without recreating `@mlxts/hub`, GGUF
interoperability enters through a thin custom MLX bridge instead of a repo-owned
format parser, and the new data surfaces remain plain TypeScript plus explicit
MLX array construction at the batch boundary.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/ffi/lib.ts`
- `packages/core/src/index.ts`
- `packages/core/src/io.ts`
- `packages/core/src/io-gguf.ts`
- `packages/core/src/quantization.ts`
- `packages/data/src/chat.ts`
- `packages/data/src/collation.ts`
- `packages/data/src/dataset.ts`
- `packages/data/src/index.ts`
- `packages/data/src/jsonl.ts`
- `packages/data/src/preference.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/lora-linear.ts`
- `packages/nn/src/module.ts`
- `packages/nn/src/quantized-linear.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/infrastructure/generation-defaults.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/load.ts`
- `packages/transformers/src/quantize.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

- `quantizedMatmul()` was added as a first-class core primitive rather than
  hidden behind a dense dequantize fallback, and the tensor-lifetime tracked-op
  list now includes `quantize`, `dequantize`, and `quantizedMatmul` so the
  static gate keeps covering the new quantized path.
- `QuantizedLinear.forward()` keeps the packed-weight kernel output visible as a
  named local before any optional bias add, instead of nesting the quantized
  matmul inside another tensor-producing call.
- `QuantizedLinear.toLinear()` dequantizes once into a named dense weight and
  then transfers ownership into the returned `Linear` module; there is no
  hidden aliasing back into the quantized layer.
- `LoRALinear.forward()` keeps base output, dropout output, low-rank product,
  delta, and scaled delta as explicit `using` locals. That keeps adapter math
  readable and prevents anonymous intermediates from being buried inside one
  large expression.
- `Module.replaceChild()` now has explicit ownership semantics: it replaces only
  direct child modules, returns the previous child, and never auto-disposes it.
  That is important because LoRA wrappers can continue owning a base layer after
  replacement.
- `loadCausalLM()` prepares quantized placeholders before weight assignment and
  still keeps checkpoint tensor ownership explicit: ignored weights are skipped
  before assignment, assigned tensors transfer into the module tree, and loader
  errors still dispose the partially built model.
- `quantizePretrainedSnapshot()` keeps shard rewriting bounded to one shard at a
  time. Each shard accumulates explicit `rewritten` tensor locals, saves the
  shard, and then frees every retained tensor in a `finally` block before the
  next shard begins.
- `loadGguf()` uses MLX's native `load_gguf()` path behind a custom bridge, then
  copies named map entries into owned `MxArray` handles on the TypeScript side
  before freeing the native map container. GGUF metadata is serialized to JSON
  on the native side so TypeScript never has to hold variant-typed metadata
  ownership through Bun FFI.
- `generateTokens()` and `generateText()` now merge checkpoint generation
  defaults with explicit options and tokenizer EOS ids before decode begins, so
  EOS handling stays a simple named local set instead of model-specific stop
  logic being scattered through the decode loop.
- `collateTokenSupervisionBatch()` and `collatePreferenceBatch()` keep the data
  boundary explicit: host-side typed arrays are filled first, then wrapped into
  MLX arrays with named `using` locals before reshape. There are no nested
  tensor constructors hiding temporary ownership in the collation path.

## Memory / Performance Evidence

- `bun run typecheck` passes across the full workspace after the Phase 8/9
  additions.
- `bun run check:coverage` passes across the canonical package stack and the
  temporary `packages/nanogpt/` fixture. The new packages cleared the gate with
  `@mlxts/quantize` at `99.01%` lines, `@mlxts/lora` at `95.38%`, and
  `@mlxts/align` at `97.27%`.
- Focused runtime validation for this continuation also passes:
  `bun run --filter '@mlxts/core' build:native`,
  `bun run --filter '@mlxts/core' typecheck`,
  `bun run --filter '@mlxts/quantize' typecheck`,
  `bun run --filter '@mlxts/transformers' typecheck`,
  `bun test packages/core/src/io-gguf.test.ts packages/quantize/src/gguf.test.ts`,
  and
  `bun test packages/transformers/src/infrastructure/generation-defaults.test.ts packages/transformers/src/load.test.ts`.
- Focused quantization correctness coverage now includes a direct
  `quantizedMatmul` parity test against dense matmul with dequantized weights.
- Focused loader coverage now includes checkpoint-driven quantized module setup
  from model config metadata before safetensor assignment and shard-wise dense
  snapshot rewriting into loadable quantized checkpoints.
- Focused LoRA coverage exercises dense merge, QLoRA merge, explicit path
  targeting, already-wrapped selection behavior, and adapter save/load
  round-trips.
- Focused data and alignment coverage exercises JSONL loading, padding/collation
  behavior, chat-template supervision building, SFT training, and DPO training.
- This review still does not include generation throughput benchmarks because
  the current `generation.ts` diff changes option resolution and EOS/default
  selection, not the token-decode kernel count, cache update logic, or sampler
  execution structure. A hot-path benchmark refresh becomes mandatory only if
  the decode loop itself changes.

## Independent Review

Implementation is being done by Codex. Independent review is still required
before this milestone should be considered fully closed.

## Remaining Risks / Follow-ups

- GGUF tensor import is now available through MLX-native loading, but it still
  inherits MLX's current behavior: `Q4_0`, `Q4_1`, and `Q8_0` stay directly
  quantized, while formats like `Q4_K_M` and `Q6_K` dequantize to dense
  `float16` on load. Efficient non-MLX-native GGUF quant formats remain future
  work if we want them to preserve packed-weight memory savings end to end.
- `loadCausalLM()` now honors checkpoint quantization metadata, but broader
  checkpoint-specific generation defaults remain separate work. In particular,
  Gemma-family chat behavior should still be evaluated through the full
  chat-template and generation-default path rather than blamed on quantization
  or LoRA by default.
- Alignment recipes now exist as reusable package surfaces, but longer
  convergence evidence and recipe-level operator UX still belong to later
  training and serving milestones.
