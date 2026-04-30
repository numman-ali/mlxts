# Stable Diffusion Config Parsing

## Summary

Added package-native config parsing for Stable Diffusion VAE and UNet
components in `@mlxts/diffusion`. The tranche translates Diffusers
`vae/config.json` and `unet/config.json` into stable TypeScript config shapes,
normalizes scalar-or-per-block UNet fields, and rejects known unsupported
semantic knobs before model construction exists.

This tranche does not construct VAE or UNet modules, load safetensor payloads,
run text conditioning, denoise latents, or generate images.

## Files Reviewed

- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/families/stable-diffusion/config-parsing.ts`
- `packages/diffusion/src/families/stable-diffusion/config.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py`
  defines the AutoencoderKL config fields used here: channels, block types,
  block widths, latent channels, sample size, scaling factor, quant conv flags,
  and latent-shift/stat fields.
- `.reference/diffusers/src/diffusers/models/unets/unet_2d_condition.py`
  defines the UNet2DConditionModel config fields used here: channels, block
  widths, block types, layers per block, attention head dimensions,
  cross-attention dimensions, text-time addition embedding fields, and
  unsupported class/encoder/cross-attention variants.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/config.py`
  identifies the package-native target shapes for Autoencoder and UNet configs.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  confirms Diffusers-to-MLX translation for VAE latent channels, scaling
  factor, UNet repeated scalar fields, and SDXL text-time embedding fields.

## Tensor Lifetime Audit

This tranche reads JSON metadata and returns plain config objects. It does not
create `MxArray` values and does not add tensor-producing expressions.

## Memory / Performance Evidence

No benchmark is required. Config translation is host-side validation and does
not touch denoising hot paths, VAE decode paths, scheduler stepping, serving, or
model weight materialization.

Focused tests cover VAE parsing, UNet scalar normalization, SDXL text-time
conditioning fields, local snapshot manifest config loading, unsupported VAE
latent-math knobs, unsupported UNet semantic variants, non-Stable-Diffusion
manifests, and malformed component JSON.

## Independent Review

Hypatia completed a read-only second pass on the next Phase 10 diffusion
boundary and recommended a config-only Stable Diffusion tranche before model
construction, tokenizer/text-encoder loading, or image generation. The
implementation keeps the scope at config translation and leaves model
construction for the next reviewed tranche.

## Remaining Risks / Follow-ups

- VAE and UNet module construction remain unimplemented.
- Weight-path mapping and safetensor assignment remain follow-on tranches.
- Flux config parsing remains blocked on FlowMatch scheduler support and the
  Flux transformer/VAE reference audit.
- Diffusers `up_block_types` order is preserved as checkpoint truth; the model
  construction tranche owns any traversal-order mapping needed by the package
  module implementation.
