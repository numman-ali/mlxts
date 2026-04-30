# T5 Encoder Conditioning

## Summary

Added a transformer-owned T5 encoder surface for FLUX conditioning and future
encoder-decoder proof paths. The tranche adds config parsing, encoder-only model
execution, relative-position self-attention, gated and non-gated T5 feed-forward
layers, explicit weight mapping, and a local snapshot loader.

T5 stays outside the `CausalLM` registry. Diffusion still consumes prepared
conditioning tensors and does not import `@mlxts/transformers`.

## Files Reviewed

- `packages/transformers/src/families/t5/attention.ts`
- `packages/transformers/src/families/t5/block.ts`
- `packages/transformers/src/families/t5/config.ts`
- `packages/transformers/src/families/t5/mlp.ts`
- `packages/transformers/src/families/t5/model.ts`
- `packages/transformers/src/families/t5/types.ts`
- `packages/transformers/src/families/t5/weights.ts`
- `packages/transformers/src/families/t5/load.ts`
- `packages/transformers/src/index.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py`
  confirms FLUX uses CLIP pooled embeddings plus T5 encoder hidden states, with
  T5 tokenization padded/truncated to `max_sequence_length` and no
  Stable-Diffusion-style negative prompt CFG in the base FLUX path.
- `.reference/transformers/src/transformers/models/t5/modeling_t5.py`
  confirms `T5EncoderModel` owns shared token embeddings, an encoder stack,
  RMS-style layer norm without bias, first-layer relative attention bias,
  bias-free q/k/v/o projections, and gated `DenseReluDense` variants.
- `.reference/mlx-examples/flux/flux/t5.py` confirms the MLX-native shape:
  bidirectional relative-position bias, `scale=1.0` SDPA, RMSNorm, T5 v1.1
  gated activations, and encoder-only hidden-state output.

`.reference/diffusers` was fetched before audit and was even with
`origin/main`. `.reference/transformers` was read-only because its checkout has
pre-existing deleted test files.

## Tensor Lifetime Audit

The T5 attention path names q/k/v projections, reshapes, transposes, relative
position bucket embeddings, SDPA output, merged output, and final projection as
visible `using` bindings. The model run path owns the shared position bias for
the whole encoder pass and frees it in `finally`.

Returned `T5EncoderModelOutput` tensors are caller-owned. `hiddenStates` are
retained explicitly and released by `disposeT5EncoderModelOutput()`.

Weight loading assigns one safetensor at a time. Unmapped tensors are freed
immediately; assignment failures free the candidate tensor before rethrowing.

## Memory / Performance Evidence

- `bun test packages/transformers/src/families/t5` passed: 16 tests.
- `bun run --filter @mlxts/transformers typecheck` passed.
- `bench:generation` was not run: the tranche adds an encoder-only T5 surface
  outside `CausalLM` and does not touch decoder generation routing.
- `bench:generation:parity` was not run for the same reason; no paired
  autoregressive decode behavior changed.

No generation hot path changed, so no Qwen/Gemma decode benchmark was required.
The new T5 encoder path is runtime-sensitive and will need real checkpoint
evidence before FLUX prompt-conditioning is advertised as product-complete.

## Independent Review

Linnaeus completed a read-only second-opinion reference audit for the T5/FLUX
seam. The review confirmed the bounded scope, keeping T5 outside `CausalLM`,
mapping `shared.weight` into one embedding slot, using T5 scale `1.0`, reusing
first-layer bidirectional relative-position bias, and leaving prompt-level
FLUX composition outside `@mlxts/diffusion`.

The review called out a direct additive-mask test for T5 attention. This tranche
adds that test in `packages/transformers/src/families/t5/model.test.ts`.

## Remaining Risks / Follow-ups

- `@mlxts/tokenizers` already loads SentencePiece, but FLUX prompt composition
  still needs a fixed-length T5 text-input helper or example-level conditioning
  composer before real prompts can feed the FLUX tensor pipeline.
- The FLUX transformer backbone is still unimplemented; this tranche only
  unlocks the T5 text-encoder side of conditioning.
