# Stable Diffusion VAE Construction

## Summary

Added Stable Diffusion AutoencoderKL module construction in `@mlxts/diffusion`.
The tranche builds the VAE encoder, decoder, residual blocks, mid self-attention,
upsample/downsample modules, quant convs, and a disposable posterior object.
The VAE keeps NHWC tensor semantics, exposes unscaled encode/decode behavior,
and leaves checkpoint weight mapping and pipeline latent scaling to follow-up
tranches.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/autoencoder.ts`
- `packages/diffusion/src/families/stable-diffusion/autoencoder-blocks.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/mlx-examples/stable_diffusion/stable_diffusion/vae.py` confirms
  the MLX VAE shape convention: NHWC tensors, encoder
  `conv_in -> down blocks -> resnet/attention/resnet mid -> norm/silu/conv_out`,
  decoder `conv_in -> mid -> up blocks -> norm/silu/conv_out`, bottom/right
  pad before stride-2 downsample, nearest upsample before 3x3 convolution, and
  single-head spatial attention in the VAE mid block.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/unet.py` confirms
  the ResNet block shape used by the MLX VAE path: GroupNorm, SiLU, 3x3
  convolutions, residual add, and a 1x1 shortcut when channels change.
- `.reference/diffusers/src/diffusers/models/autoencoders/vae.py` and
  `.reference/diffusers/src/diffusers/models/unets/unet_2d_blocks.py` confirm
  Diffusers AutoencoderKL construction: encoder downsample on all but the final
  block, decoder `layers_per_block + 1` resnets per up block, `double_z=true`
  posterior moments, `quant_conv`, `post_quant_conv`, and mid attention when
  `mid_block_add_attention=true`.

## Tensor Lifetime Audit

All module forwards keep tensor-producing intermediates in named locals with
lexical disposal or explicit handoff. Loop-carried hidden states free the
previous owned value after each block and free the current value on error.
The posterior object owns its split mean/log-variance tensors and implements
`Symbol.dispose`; `mode()` and `sample()` return fresh tensors so callers can
dispose posterior state independently.

## Memory / Performance Evidence

No benchmark is required for this construction-only tranche. The implementation
uses the native `Conv2d`, `GroupNorm`, `Linear`, matmul, softmax, pad, and
reshape surfaces already in the package stack. It does not add host-side tensor
emulation or new runtime strategy flags.

Focused validation passed:

- `bun test packages/diffusion/src/families/stable-diffusion/autoencoder.test.ts`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run validate`
- `bun run build`

## Independent Review

Hume completed a read-only reference pass against the local MLX and Diffusers
repos. The review found no construction blocker, confirmed the NHWC block
ordering and mid-attention requirements, and recommended keeping VAE encode and
decode unscaled so pipeline/helper code owns latent scaling. That recommendation
is reflected in the public API.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

- Stable Diffusion VAE weight-path mapping and safetensor assignment still need
  a dedicated tranche.
- Pipeline latent scaling is intentionally outside the VAE module contract.
- Stochastic pipeline sampling still needs the later pipeline orchestration
  tranche; this tranche only exposes the posterior primitive.
