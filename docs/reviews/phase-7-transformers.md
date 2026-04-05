# Runtime Review: Phase 7 Tokenizers, Hub, and Transformers

## Summary

This review covers the runtime-sensitive production changes for Phase 7's
pretrained-loading surface: the new `@mlxts/hub` snapshot and artifact
inspection package, the expanded `@mlxts/tokenizers` package, and the rebuilt
`@mlxts/transformers` decoder-loading and generation stack, including the core
large-safetensor loading fix that made public Phi-4-scale checkpoints viable.

The implementation keeps the Phase 7 contract intentionally narrow and explicit:
dense text decoders only, explicit family registry, standalone generation
helpers, shard-iterator weight loading, and Gemma RMSNorm offset handling kept
local to transformers. The current dense family surface now covers the original
LLaMA/Mistral/Gemma path plus Phi-3, Gemma 3 text, Gemma 4 dense text
(`gemma4_text` plus top-level `gemma4` language-model extraction), and the
text-decoder portion of Mistral 3 without introducing speculative MoE or
multimodal contract widening.

This closeout also includes the merge-hardening cleanup that landed after the
main performance pass: `tekken.json` now ships in the default snapshot include
set for Mistral 3, prefill cache-state views are freed explicitly after eval,
and the shared llama-like RMSNorm path now caches its effective `1 + weight`
tensor the same way Gemma 3 already did. The final cleanup also keeps Gemma 3's
layer-shape bookkeeping on private fields and only instantiates Gemma 4's
compiled logit-softcap closure when the checkpoint actually enables softcapping.

## Files Reviewed

- `packages/core/src/device.ts`
- `packages/core/src/ffi/lib.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/io-safetensors-format.ts`
- `packages/core/src/io-safetensors.ts`
- `packages/core/src/io.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/arithmetic.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/reduction.ts`
- `packages/core/src/ops/shape.ts`
- `packages/hub/src/gguf.ts`
- `packages/hub/src/http.ts`
- `packages/hub/src/index.ts`
- `packages/hub/src/inspect.ts`
- `packages/hub/src/paths.ts`
- `packages/hub/src/patterns.ts`
- `packages/hub/src/snapshot.ts`
- `packages/hub/src/types.ts`
- `packages/hub/src/weights.ts`
- `packages/nn/src/embedding.ts`
- `packages/nn/src/linear.ts`
- `packages/nn/src/activations.ts`
- `packages/tokenizers/src/bpe-base.ts`
- `packages/tokenizers/src/bpe-load.ts`
- `packages/tokenizers/src/bpe.ts`
- `packages/tokenizers/src/byte-level.ts`
- `packages/tokenizers/src/char.ts`
- `packages/tokenizers/src/errors.ts`
- `packages/tokenizers/src/index.ts`
- `packages/tokenizers/src/load.ts`
- `packages/tokenizers/src/sentencepiece-proto.ts`
- `packages/tokenizers/src/sentencepiece.ts`
- `packages/tokenizers/src/tekken.ts`
- `packages/tokenizers/src/tokenizer.ts`
- `packages/transformers/src/auto.ts`
- `packages/transformers/src/families/gemma/config.ts`
- `packages/transformers/src/families/gemma/weights.ts`
- `packages/transformers/src/families/gemma3/attention.ts`
- `packages/transformers/src/families/gemma3/block.ts`
- `packages/transformers/src/families/gemma3/config.ts`
- `packages/transformers/src/families/gemma3/mlp.ts`
- `packages/transformers/src/families/gemma3/model.ts`
- `packages/transformers/src/families/gemma3/norm.ts`
- `packages/transformers/src/families/gemma3/types.ts`
- `packages/transformers/src/families/gemma3/weights.ts`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/config.ts`
- `packages/transformers/src/families/gemma4/mlp.ts`
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/families/gemma4/norm.ts`
- `packages/transformers/src/families/gemma4/rope.ts`
- `packages/transformers/src/families/gemma4/types.ts`
- `packages/transformers/src/families/gemma4/weights.ts`
- `packages/transformers/src/families/llama-like/attention.ts`
- `packages/transformers/src/families/llama-like/block.ts`
- `packages/transformers/src/families/llama-like/mlp.ts`
- `packages/transformers/src/families/llama-like/model.ts`
- `packages/transformers/src/families/llama-like/norm.ts`
- `packages/transformers/src/families/llama-like/types.ts`
- `packages/transformers/src/families/llama/config.ts`
- `packages/transformers/src/families/llama/weights.ts`
- `packages/transformers/src/families/mistral/config.ts`
- `packages/transformers/src/families/mistral/weights.ts`
- `packages/transformers/src/families/mistral3/config.ts`
- `packages/transformers/src/families/mistral3/weights.ts`
- `packages/transformers/src/families/phi/config.ts`
- `packages/transformers/src/families/phi/weights.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/cache.ts`
- `packages/transformers/src/infrastructure/config-parsing.ts`
- `packages/transformers/src/infrastructure/generation-helpers.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/infrastructure/sampling.ts`
- `packages/transformers/src/infrastructure/weight-assignment.ts`
- `packages/transformers/src/load.ts`
- `packages/transformers/src/registry.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

- `@mlxts/hub` and `@mlxts/tokenizers` stay in pure file/JSON/string space and do
  not introduce native tensor ownership. The runtime review there focused on
  file filtering, manifest parsing, GGUF header parsing, and tokenizer file
  selection rather than array lifetime hazards.
- `@mlxts/transformers` keeps tensor ownership visible in hot paths:
  `LlamaLikeDecoderBlock`, `LlamaLikeAttention`, and `LlamaLikeModel` use named
  `using` bindings for disposable intermediates instead of nesting tensor
  producers inside larger expressions.
- The Phi-3 packed-projection path keeps the extra tensor ownership explicit:
  `LlamaLikeAttention.projectQueryKeyValue()` returns owned query/key/value
  slices from the packed `qkvProjection`, and `LlamaLikeMLP.projectGateAndValue()`
  returns owned gate/value tensors from `gateUpProjection`, both of which are
  freed in the caller after the forward path completes.
- Independent review identified one remaining lifetime cleanup in
  `LlamaLikeMLP.forward`: the `gelu_pytorch_tanh` branch previously fed
  `geluPytorchTanh(gate)` directly into `multiply(...)`. That path now uses a
  named `using geluResult` binding before the multiply so every disposable
  intermediate stays visible.
- Independent review also flagged the decoder block's `mlp` field for being
  typed too concretely as `LlamaLikeMLP`. The field now uses a minimal
  forward-capable module interface so the future MoE block swap stays a clean
  structural replacement rather than a concrete subclass dependency.
- `loadCausalLM()` owns checkpoint tensors explicitly: ignored or unmapped shard
  tensors are freed immediately, assigned tensors transfer ownership into the
  model tree, and error paths dispose the partially built model.
- `@mlxts/core` no longer materializes an entire safetensors shard through one
  `arrayBuffer()` call before parsing it. `loadSafetensors()` now reads the
  header first and then creates tensor views one slice at a time, and
  `iterateSafetensors()` keeps tensor ownership localized to the currently
  yielded entry.
- `iterateSafetensorWeights()` now supports pre-load filtering, which lets the
  Phase 7 loaders skip ignored weights before they become `MxArray`s. That is
  the key change that prevents top-level multimodal Gemma 4 checkpoints from
  loading audio or vision tensors just to throw them away.
- `LayerPatternKVCache` keeps the mixed Gemma 3 retention policy explicit per
  layer instead of hiding it behind attention special cases: full-attention
  layers retain all prior keys and values, while sliding layers trim on update
  with the same visible ownership transitions as `SlidingWindowKVCache`.
- The new Gemma 4 dense text path keeps its extra ownership points visible
  instead of burying them in family-specific helpers:
  `Gemma4TextAttention.buildFreshKeyValues()` binds value-head creation
  explicitly before q/k/v normalization, `Gemma4TextDecoderBlock.run()` frees
  retained key/value views on the plain `forward()` path, and
  `Gemma4TextModel.runLayers()` owns the retained shared-KV snapshots that later
  layers borrow for the config-driven KV-sharing path.
- Gemma 4's proportional full-attention RoPE stays local to transformers via
  `Gemma4ProportionalRoPE`, which owns its precomputed frequency tensor as a
  private field and disposes it explicitly rather than smuggling non-parameter
  arrays onto the module tree.
- Gemma 4's optional per-layer input stream also keeps all disposable arrays
  visible: `createPerLayerInputs()` names the embedding, projection, reshape,
  normalization, and split intermediates, and the model frees the resulting
  per-layer slices in a single finally block after the decoder stack finishes.
- `createCausalMask()` now has an explicit windowed-causal mode for local
  attention prefill. The mask is still built out of visible intermediate arrays
  rather than nested tensor expressions, so the static lifetime audit can follow
  it line by line.
- The Gemma 3 text blocks keep all local tensor lifetimes visible across q/k
  norm, per-layer RoPE, and the four-norm residual structure. No intermediate
  attention or MLP activations are hidden inside nested arithmetic.
- `generateStep()` frees caller-owned temporary input tensors when token ids are
  passed as plain arrays, and cache objects remain explicitly disposable.
- `prefillPromptCache()` now frees the retained cache-state views returned by
  `cacheStateArrays(cache)` immediately after `mxEval(...)` completes, which
  closes the last bounded retained-view leak the independent review found in the
  chunked prefill path.
- `bun run check:tensor-lifetimes` passes on the Phase 7 tree after the final
  transformer changes.

## Memory / Performance Evidence

- Repo-level quality gates now enforce `95%` lines / `90%` functions coverage
  across the canonical package stack and the temporary `packages/nanogpt/`
  validation fixture via `bun run check:coverage`.
- `@mlxts/transformers` has explicit cache-correctness coverage:
  cached and uncached continuation logits match for the same prefix, and cached
  vs uncached token generation produces the same deterministic output for the
  tiny fixture snapshots.
- The dense family loader surface now has tiny-snapshot coverage for the new
  variants added in this pass: Phi-3 (`model_type: "phi3"`), Gemma 3 text
  (`"gemma3_text"`), Gemma 4 dense text (`"gemma4_text"` plus top-level
  `"gemma4"` with `model.language_model.`-prefixed weights), and the
  text-decoder portion of Mistral 3 (`"mistral3"` with `language_model.`-prefixed
  weights).
- The Phase 7 performance surface now has two first-class benchmark modes:
  `bun run bench:generation` for synthetic throughput canaries and
  `bun run bench:generation:parity` for MLX-LM comparison. Shared benchmark
  code lives under `packages/transformers/scripts/benchmark-*.ts`, and
  recorded baselines now live in the two-surface `benchmarks/baselines.json`.
- The final hot-path performance fix in this pass is a repo-owned native core
  primitive, `geluApprox`, added because MLX-C does not expose `gelu_approx`
  directly. That primitive now replaces the previous compiled closure path in
  Gemma 3 MLPs, Gemma 4 dense-text MLPs, Gemma 4 per-layer input gating, and
  the shared llama-like GELU branch.
- Fresh three-trial parity measurements on this M4 Max 64 GB machine
  (`prompt_tokens=1024`, `generation_tokens=128`) now show mlxts at or above
  the paired MLX-LM reference runs captured on the same day:
  `mlx-community/Llama-3.2-1B-Instruct-bf16` measured
  `3619.8 prompt tok/s`, `55.4 decode tok/s`, `3.089 GB peak` in mlxts versus
  `2856.5`, `50.2`, `2.937 GB` in MLX-LM;
  `google/gemma-3-1b-it` measured `3229.9 prompt tok/s`,
  `57.9 decode tok/s`, `2.758 GB peak` in mlxts versus
  `2932.2`, `50.5`, `2.683 GB` in MLX-LM;
  `microsoft/Phi-4-mini-instruct` measured `1973.8 prompt tok/s`,
  `57.5 decode tok/s`, `8.714 GB peak` in mlxts versus
  `1494.5`, `39.5`, `8.421 GB` in MLX-LM.
- Fresh three-trial synthetic throughput baselines now recorded in
  `benchmarks/baselines.json` are:
  Llama 3.2 1B at `6657.4 prompt tok/s` / `169.9 decode tok/s`,
  Gemma 3 1B at `7089.6` / `163.6`,
  Phi-4 mini at `2032.6` / `57.0`.
- The decode hot path now holds the core performance invariants the repo cares
  about: one eval per token in steady state, GPU-side sampling, dedicated
  generation stream ownership, recommended working-set wired limit, and static
  cache writes instead of per-token dynamic index tensors.
- Automated validation currently exercised during this review:
  `bun run build`, `bun run docs:api`, `bun run typecheck`,
  `bun run check:tensor-lifetimes`, package-level tests for hub/tokenizers/train/transformers,
  and `bun run check:coverage`.

## Independent Review

This artifact was prepared by the implementation author and then independently
reviewed by a separate Claude agent. That follow-up review specifically checked
the tensor-lifetime discipline in the rebuilt llama-like path, called out the
missing named GELU intermediate in `LlamaLikeMLP.forward`, and identified the
overly concrete `mlp: LlamaLikeMLP` field typing in `LlamaLikeDecoderBlock`.
Both findings are now fixed in the reviewed tree.

## Remaining Risks / Follow-ups

- Gemma 4 dense text loading is now implemented directly on our primitives, but
  the full conditional-generation wrapper remains intentionally out of this
  Phase 7 review. The current implementation only loads the
  `model.language_model.*` portion of top-level Gemma 4 checkpoints and ignores
  vision/audio weights by design.
- The public `google/gemma-4-E2B-it` checkpoint still does not complete a
  practical local acceptance run on this 64 GB machine. A single-model
  load-and-generate attempt stayed alive for over a minute and then exited
  without returning a generation result, so the dense Gemma 4 path is
  implemented but not yet signed off as locally production-ready for that
  checkpoint.
- The dense Gemma 4 path currently covers the observed public non-MoE variants:
  the 31B-style path (full/sliding mix, `attention_k_eq_v`, no per-layer inputs)
  and the E2B-style path (per-layer inputs, KV sharing, double-wide shared MLP).
  MoE-enabled Gemma 4 variants remain explicit Phase 7e work.
- The current Mistral 3 support is intentionally the text-decoder portion only:
  `loadCausalLM()` extracts `text_config`, strips the `language_model.` prefix,
  and ignores vision/projector weights. That is the correct Phase 7 behavior,
  but the full multimodal wrapper still belongs to the later multimodal work.
- The newest public Mistral 3 checkpoints ship `tekken.json`, and Tekken
  support is now implemented in `@mlxts/tokenizers`. However, a single-model
  local acceptance run for `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
  still did not complete within the bounded acceptance window on this machine,
  so latest-family Mistral generation is not yet signed off as a practical
  local run.
- The real-model LLaMA parity export and cache benchmark are intentionally
  offline/manual because they require a local MLX model snapshot and MLX Python.
  The harnesses are in place, but the final Apple Silicon numbers still need to
  be recorded against a real model snapshot.
- `CharTokenizer.encode()` now iterates by Unicode code point rather than
  UTF-16 code unit. That is the correct runtime behavior, but it is the one
  behavior change in the tokenizer expansion that deserves a quick human sanity
  check against any downstream assumptions that previously relied on code-unit
  indexing.
- `packages/nanogpt/` now satisfies the same coverage gate as the extracted
  packages, but it remains a temporary validation fixture rather than a
  long-term product surface.
