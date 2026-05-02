# Qwen-Image Reference Latent Prep Review

## Summary

This tranche adds the package-owned image-conditioning foundation needed by
Qwen-Image Edit, Edit Plus, img2img, inpaint, and future control paths without
claiming those denoising modes yet.

`QwenImageAutoencoderKL` now exposes deterministic `encodeRaw()` over the
posterior mode. `encodeQwenImageLatents()` applies Qwen latent mean/std
normalization and Diffusers-compatible 2x2 latent packing. `prepareQwenImageReferenceLatents()`
builds multi-reference packed latent sequences and records each reference
segment's RoPE shape so later edit denoising can compose target plus reference
segments explicitly.

The runtime fence remains deliberate: edit generation still needs processor
reference-image conditioning, multi-segment RoPE and denoiser input/output
slicing, and `zero_cond_t` modulation.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/autoencoder.ts`
- `packages/diffusion/src/families/qwen-image/conditioning.ts`
- `packages/diffusion/src/families/qwen-image/latent-stats.ts`
- `packages/diffusion/src/families/qwen-image/latents.ts`
- `packages/diffusion/src/families/qwen-image/pipeline.ts`
- `packages/diffusion/src/families/qwen-image/autoencoder.test.ts`
- `packages/diffusion/src/families/qwen-image/conditioning.test.ts`
- `packages/diffusion/src/families/qwen-image/latents.test.ts`
- `packages/diffusion/src/families/qwen-image/pipeline.test.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

Diffusers Qwen-Image edit paths encode reference images through the Qwen VAE,
retrieve posterior latents, normalize with `latents_mean` / `latents_std`, pack
the latent grid, concatenate target and reference latent sequences for edit
denoising, and pass target/reference `img_shapes` into the transformer.

Relevant local references:

- `.reference/diffusers/src/diffusers/modular_pipelines/qwenimage/encoders.py`
- `.reference/diffusers/src/diffusers/modular_pipelines/qwenimage/denoise.py`
- `.reference/diffusers/src/diffusers/modular_pipelines/qwenimage/before_denoise.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py`
- `.reference/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py`
- `.reference/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py`

## Tensor Lifetime Audit

The new tensor-producing surfaces keep disposable intermediates visible:
`encodeRaw()` frees unused posterior log-variance, `encodeQwenImageLatents()`
binds raw latents, mean, std, shifted, and normalized tensors before packing,
and `prepareQwenImageReferenceLatents()` frees every retained packed segment
after returning either a retained single segment or a concatenated sequence.

`pipeline.ts` now shares latent-stat tensor creation with the encode path
instead of carrying a private duplicate helper. Base text-to-image denoising
control flow is unchanged.

## Memory / Performance Evidence

No generation hot path, scheduler stepping, or serving path changed. The new
helpers execute only when an image-conditioning caller explicitly encodes
reference images. Base Qwen-Image text-to-image generation still uses
`decodeQwenImageLatents()` and the existing denoising loop.

Focused tests:

```bash
bun test packages/diffusion/src/families/qwen-image/autoencoder.test.ts packages/diffusion/src/families/qwen-image/latents.test.ts packages/diffusion/src/families/qwen-image/conditioning.test.ts packages/diffusion/src/families/qwen-image/pipeline.test.ts
```

Result: 24 pass, 0 fail.

Additional focused gates:

```bash
bun run typecheck
bun run check:tensor-lifetimes
```

Both passed.

Full validation:

```bash
bun run validate
```

Result: passed.

## Independent Review

Wegener the 2nd performed a read-only second pass over the local Qwen-Image
runtime and Diffusers references. The review rejected `zero_cond_t` alone as a
product-real tranche because upstream edit semantics bind it to reference
latents plus multi-segment `img_shapes`, and recommended normalized VAE
reference-latent preparation as the smallest honest next capability.

## Remaining Risks / Follow-ups

- Qwen-Image Edit/Edit Plus still needs processor reference-image prompt
  conditioning with the Qwen2VL image template.
- The transformer still accepts a single `imageShape`; edit denoising needs
  multi-segment target/reference RoPE and output slicing.
- `zero_cond_t` runtime modulation remains unsupported until the multi-segment
  denoiser path exists.
- Img2img, inpaint, ControlNet, layered, LoRA, quantized sidecars, and real edit
  checkpoint evidence remain separate Phase 10 tranches.
