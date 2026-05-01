# Qwen-Image Transformer Foundation Runtime Review

## Summary

Qwen-Image now has a package-owned base transformer runtime over prepared
Diffusers tensors: packed image latents, Qwen text hidden states, optional text
mask, timestep, and image RoPE shape. The tranche also loads Diffusers
transformer safetensors into the package parameter tree.

This does not claim full Qwen-Image text-to-image generation yet. Text
encoding, prompt orchestration, classifier-free guidance, denoising pipeline
assembly, and real generated image evidence remain separate tranches.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/attention.ts`
- `packages/diffusion/src/families/qwen-image/blocks.ts`
- `packages/diffusion/src/families/qwen-image/embeddings.ts`
- `packages/diffusion/src/families/qwen-image/tensor-utils.ts`
- `packages/diffusion/src/families/qwen-image/transformer.ts`
- `packages/diffusion/src/families/qwen-image/weight-mapping.ts`
- `packages/diffusion/src/families/qwen-image/weights.ts`
- `packages/diffusion/src/index.ts`

## Reference Parity

Reviewed Diffusers `QwenImageTransformer2DModel` as the semantic source of
truth. The implemented path mirrors the base text-to-image components:

- timestep projection uses the Qwen/Diffusers 256-channel sinusoidal embedding
  with scale `1000`;
- image and text inputs project separately into the hidden size;
- text hidden states are RMS-normalized before text projection;
- each block uses separate image/text modulation, affine-free layer norm, joint
  non-causal attention over `[text, image]`, Q/K RMSNorm, and GELU approximate
  feed-forward layers;
- Qwen RoPE uses `[frame, height, width]` axes and scaled height/width positions
  before joint attention;
- text masks stay as masks and do not shorten the text RoPE length;
- final projection applies AdaLayerNormContinuous-style scale/shift before
  `proj_out`.

Unsupported Qwen variants are rejected deliberately:

- `guidance_embeds`
- `zero_cond_t`
- `use_additional_t_cond`
- `use_layer3d_rope`

## Tensor Lifetime Audit

New runtime helpers keep tensor-producing intermediates in visible `using`
bindings. Longer-lived block state is retained explicitly and freed in
`finally` blocks when hidden streams are replaced. Attention projections and
block outputs expose disposable tensors at clear ownership boundaries.

## Memory / Performance Evidence

This tranche adds a new runtime path and does not make performance claims.
Focused tests run tiny tensor shapes through the Qwen-Image attention, block,
transformer, and generated safetensor loading paths to exercise ownership and
shape contracts without heavy checkpoint execution.

## Validation

- `bun test packages/diffusion/src/families/qwen-image`: 52 pass, 0 fail.
- `bun run --filter '@mlxts/diffusion' typecheck`: passed.

## Independent Review

McClintock the 2nd completed a read-only second-opinion audit for this tranche.
The review recommended a correctness-first base Qwen-Image transformer runtime
over prepared tensors plus transformer weight loading, while keeping prompt
encoding, CFG, edit/layered variants, and real image proof out of scope.

## Remaining Risks / Follow-ups

- Full Qwen-Image product behavior still needs Qwen2.5-VL prompt embedding
  orchestration outside `@mlxts/diffusion`.
- Prepared-embedding denoising and final image generation need a separate
  pipeline tranche with real checkpoint evidence.
- `zero_cond_t`, layered/edit/control/inpaint paths, and layer-3D RoPE remain
  unsupported until their runtime semantics are implemented deliberately.

## Out-of-scope Drift Noticed

None.
