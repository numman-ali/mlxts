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

This review also covers the follow-on generation change that makes the
transformers package usable for real interactive checkpoint validation. The
generation surface can now reuse an external prompt cache across turns and
stream decoded text chunks through an explicit callback while keeping the
underlying token-decode loop and cache-update structure unchanged.

It also now covers the Phase 7 fidelity follow-up that surfaced while testing
Gemma-family chat checkpoints: tokenizer encoding must match inline
`added_tokens` before ordinary BPE segmentation, and prompt compilation for
chat-capable checkpoints now lives behind a shared interaction profile instead
of being duplicated in the example layer.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/ffi/lib.ts`
- `packages/core/src/index.ts`
- `packages/core/src/io.ts`
- `packages/core/src/io-gguf.ts`
- `packages/core/src/io-safetensors.ts`
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
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/generation.ts`
- `packages/tokenizers/src/bpe-added-tokens.ts`
- `packages/tokenizers/src/bpe-base.ts`
- `packages/tokenizers/src/bpe-merges.ts`
- `packages/transformers/src/infrastructure/generation-defaults.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/interaction-profile.ts`
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
- `saveSafetensors()` now writes headers and tensor payloads incrementally
  instead of concatenating one giant JavaScript buffer for the entire file.
  That keeps ordinary safetensors writes readable while removing a major
  large-file peak-memory hazard from the core I/O layer.
- `inspectSafetensors()` and `iterateSafetensorByteChunks()` let higher layers
  inspect shard metadata and copy raw tensor payloads without bridging those
  bytes through temporary `MxArray` ownership. That is important for giant
  copied tensors like Gemma 4's per-layer embeddings.
- `quantizePretrainedSnapshot()` no longer accumulates one `Record<string,
  MxArray>` for the whole rewritten shard. It now plans output tensor metadata
  up front, raw-copies non-quantized tensors in bounded byte chunks, and only
  caches one quantized source tensor's output bytes at a time while its
  `.weight`, `.scales`, and optional `.biases` entries are emitted.
- `loadGguf()` uses MLX's native `load_gguf()` path behind a custom bridge, then
  copies named map entries into owned `MxArray` handles on the TypeScript side
  before freeing the native map container. GGUF metadata is serialized to JSON
  on the native side so TypeScript never has to hold variant-typed metadata
  ownership through Bun FFI.
- `generateTokens()` and `generateText()` now merge checkpoint generation
  defaults with explicit options and tokenizer EOS ids before decode begins, so
  EOS handling stays a simple named local set instead of model-specific stop
  logic being scattered through the decode loop.
- `BPETokenizer.encodeWithOffsets()` now splits raw text around inline
  `added_tokens` before byte-level or sentencepiece segmentation. That keeps
  Gemma-style turn markers and byte-level control tokens visible as owned named
  chunks instead of being buried inside ordinary token-piece fallback logic.
- The added-token matcher and BPE merge helpers were split into dedicated
  internal helpers so `bpe-base.ts` stays under the repo line-limit gate
  without hiding ownership or prompt-compilation logic behind a monolithic
  tokenizer file.
- `generateTokens()` can now accept an explicitly owned prompt cache, but it
  still keeps ownership visible in code: internally created caches stay scoped
  to the helper, externally provided caches are never auto-disposed, and the
  option combination `cache + useCache: false` throws immediately instead of
  leaving ambiguous ownership or partially warmed cache state.
- `generateTextStream()` is callback-based rather than iterator-based on
  purpose. That keeps the existing dedicated generation stream scope,
  asynchronous MLX eval scheduling, and wired-memory limit logic in the same
  synchronous decode path instead of splitting ownership across a long-lived JS
  iterator object.
- `InteractionProfile.compileMessages()` now owns chat-template rendering plus
  tokenizer encoding in one shared place. That keeps checkpoint-specific prompt
  compilation out of examples and creates a single contract that later serving
  adapters can reuse without re-implementing model-family prompt logic.
- `Gemma4TextModel.createPerLayerInputs()` now scales per-layer token
  embeddings by `sqrt(hidden_size_per_layer_input)` before they are combined
  with the projected model-side per-layer inputs. That matches mlx-lm’s Gemma 4
  dense path and fixes the degraded Gemma 4 dense-generation canary without
  changing the generic generation stack.
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
  `bun test packages/core/src/io.test.ts packages/core/src/io-extra.test.ts`,
  `bun test packages/core/src/io-gguf.test.ts packages/quantize/src/gguf.test.ts`,
  and
  `bun test packages/transformers/src/infrastructure/generation-defaults.test.ts packages/transformers/src/load.test.ts`.
- The streaming generation follow-up passes `bun run typecheck` across the full
  workspace and focused transformers coverage for the new path:
  `bun test packages/transformers/src/load.test.ts packages/transformers/src/chat-template.test.ts`.
  That coverage now explicitly checks that streamed text matches buffered text,
  that a caller-owned prompt cache can be reused across turns, and that invalid
  `cache + useCache: false` combinations fail fast.
- The tokenizer and prompt-compilation follow-up now passes direct parity-style
  coverage for inline special-token encoding and shared chat compilation:
  `bun test packages/tokenizers/src/bpe.test.ts`,
  `bun test packages/transformers/src/interaction-profile.test.ts packages/transformers/src/chat-template.test.ts`.
  That coverage explicitly checks sentencepiece-style turn markers, byte-level
  inline control tokens, and compiled prompt text plus token IDs for llama-,
  mistral-, phi-, and gemma-style chat templates.
- The Gemma 4 follow-up now also passes focused regression coverage for the
  per-layer input scale path:
  `bun test packages/transformers/src/families/gemma4/model.test.ts packages/transformers/src/families/gemma4/weights.test.ts`.
  The new test captures the per-layer input tensor handed to the decoder block
  and verifies that the token-embedding branch is scaled by
  `sqrt(hiddenSizePerLayerInput)` before the standard `2^-0.5` combine step.
- Real dense-model validation now shows the repaired Gemma 4 E2B path behaving
  coherently in `examples/chat/`: the same `Hello there` prompt that
  previously devolved into repeated `wiwi...` tokens now produces
  `Hello! How can I help you today? 😊` under the greedy chat canary.
- A real cached-model export attempt on `google/gemma-4-E2B-it` now stays under
  roughly `1.1 GB` peak RSS instead of blowing up into multi-gigabyte shard
  buffering. The run still crashes inside Bun during the long-running
  large-file export path before completion, so the current blocker is runtime
  stability rather than the quantization algorithm's memory shape.
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
- The new streaming export path materially improves memory behavior for large
  checkpoints, but full end-to-end quantization of `google/gemma-4-E2B-it`
  still exposes a Bun crash in the mixed large-file copy plus quantization
  flow. Until that runtime issue is worked around or fixed upstream, the
  Gemma-4-class self-quantization proof remains partially blocked even though
  the repo-side algorithm is no longer the obvious source of the failure.
- The interactive `examples/chat/` surface now matches the intended minimal
  validation shape more closely, but it still intentionally stops short of full
  tokenizer-side chat-feature parity. If a supported checkpoint still shows
  turn-marker leakage after this change, the next fix should stay in the
  checkpoint defaults / template boundary rather than expanding into tool
  calling or `@mlxts/serve`.
- Alignment recipes now exist as reusable package surfaces, but longer
  convergence evidence and recipe-level operator UX still belong to later
  training and serving milestones.
