# Runtime Review: Phase 8 Quantize, LoRA, and Alignment Foundations

## Summary

This review covers the runtime-sensitive production changes that Phase 8 and
the first Phase 9a-adjacent quantization work add on top of the completed
Phase 7 loading surface. The core shift is from "dense-only checkpoint loading"
to a stack that can prepare quantized module placeholders before assignment,
wrap dense or quantized linear layers with LoRA adapters, and feed alignment
recipes from reusable dataset and collation contracts instead of ad hoc fixture
code.

The implementation keeps the package boundaries deliberate. Low-level MLX
quantization primitives stay in `@mlxts/core`, quantized and LoRA-aware layer
forms stay in `@mlxts/nn`, checkpoint-aware quantization setup plugs into the
existing transformers loader without recreating `@mlxts/hub`, and the new data
surfaces remain plain TypeScript plus explicit MLX array construction at the
batch boundary.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
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
- `packages/transformers/src/load.ts`

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
- Focused quantization correctness coverage now includes a direct
  `quantizedMatmul` parity test against dense matmul with dequantized weights.
- Focused loader coverage now includes checkpoint-driven quantized module setup
  from model config metadata before safetensor assignment.
- Focused LoRA coverage exercises dense merge, QLoRA merge, explicit path
  targeting, already-wrapped selection behavior, and adapter save/load
  round-trips.
- Focused data and alignment coverage exercises JSONL loading, padding/collation
  behavior, chat-template supervision building, SFT training, and DPO training.
- This review does not include generation throughput benchmarks because the
  Phase 8/9 diff does not change `generation.ts`, `sampling.ts`, cache update
  logic, or other token-decode hot paths. The one transformers runtime change
  in this slice is checkpoint-time quantized module setup in `load.ts`.

## Independent Review

Implementation is being done by Codex. Independent review is still required
before this milestone should be considered fully closed.

## Remaining Risks / Follow-ups

- GGUF tensor import and dequantization still remain future `@mlxts/quantize`
  work. This slice only prepares the module and loader structure for quantized
  checkpoints that already map onto the current autoregressive loading surface.
- `loadCausalLM()` now honors checkpoint quantization metadata, but broader
  checkpoint-specific generation defaults remain separate work. In particular,
  Gemma-family chat behavior should still be evaluated through the
  chat-template and generation-default path rather than blamed on quantization
  or LoRA by default.
- Alignment recipes now exist as reusable package surfaces, but longer
  convergence evidence and recipe-level operator UX still belong to later
  training and serving milestones.
