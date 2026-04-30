# Stable Diffusion Spatial Primitives

## Summary

Added the spatial primitive stack needed before Stable Diffusion VAE/UNet module
construction: `conv2d` and constant `pad` in `@mlxts/core`, `Conv2d` and
`GroupNorm` in `@mlxts/nn`, and Stable Diffusion-local nearest upsample plus
bottom/right padding helpers. The tranche uses existing MLX convolution and pad
operations, exports semantic layer APIs, adds focused tensor tests, and
registers new tensor-producing primitives with the tensor-lifetime guard.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/native/mlxts_core_ops.cpp`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/ops/padding.ts`
- `packages/core/src/ops/shape.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/index.ts`
- `packages/nn/src/layers/conv2d.ts`
- `packages/nn/src/layers/group-norm.ts`
- `packages/nn/src/index.ts`
- `packages/diffusion/src/families/stable-diffusion/spatial.ts`
- `scripts/runtime-sensitive-ops.ts`

## Reference Audit

- `packages/core/native/build/_deps/mlx-c-src/mlx/c/ops.h` exposes
  `mlx_conv2d` with NHWC input semantics parallel to the existing `mlx_conv1d`
  binding and exposes `mlx_pad` with explicit axes, low/high pad vectors,
  scalar pad value, mode string, and stream.
- Direct C++ execution through `mlx_conv2d` succeeded for a 2x2 NHWC input.
  Direct Bun FFI into the many-argument `mlx_conv2d` symbol segfaulted, so
  `mlxts_conv2d` is a narrow native ABI shim that takes one params buffer and
  forwards to the mlx-c operation.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/unet.py` and
  `.reference/mlx-examples/stable_diffusion/stable_diffusion/vae.py` rely on
  `nn.Conv2d`, `mx.pad`, and nearest upsampling for Stable Diffusion spatial
  blocks.
- `.reference/diffusers/src/diffusers/models/unets/unet_2d_condition.py` and
  `.reference/diffusers/src/diffusers/models/autoencoders/vae.py` confirm that
  Stable Diffusion UNet/VAE construction is convolution-heavy and requires
  spatial padding before package-owned modules can be honest.

## Tensor Lifetime Audit

`conv2d` mirrors the existing per-call `OutSlot` operation pattern through
`readResultArrayWithMetadata`; its native shim does not retain tensors beyond
the call. `pad` creates a local scalar pad value and keeps its lifetime visible
with `using` while the mlx-c call executes. `Conv2d`, `GroupNorm`, and the
Stable Diffusion spatial helpers keep every tensor intermediate in a named local
with lexical disposal. No disposable `MxArray` intermediates are hidden inside
tensor-producing calls.

## Memory / Performance Evidence

No benchmark is required. The tranche exposes native mlx-c operations already
used by MLX rather than adding host-side emulation. Focused tests exercise
channel-last `conv2d`, spatial stride/padding metadata, explicit constant
padding, `Conv2d` parameter/bias/group behavior, `GroupNorm` normalization, and
Stable Diffusion nearest upsample plus bottom/right downsample pre-padding.

Focused validation passed:

- `bun test packages/core/src/ops/ops.test.ts packages/nn/src/layers/conv2d.test.ts packages/nn/src/layers/group-norm.test.ts packages/diffusion/src/families/stable-diffusion/spatial.test.ts`
- `bun run validate`
- `bun run build`

## Independent Review

Maxwell completed a read-only second-opinion audit for the broader Phase 10
module-construction boundary and recommended this spatial-primitives
prerequisite before VAE construction, UNet construction, or weight mapping.
Bacon completed a no-edit review of the uncommitted tranche, found the initial
direct-FFI `conv2d` crash and readonly-shape type issue, and both blockers were
resolved before validation.

## Remaining Risks / Follow-ups

- VAE module construction is now the next smallest Stable Diffusion model
  tranche.
- Weight-path mapping and safetensor assignment remain blocked until the VAE
  module tree exists.
- `conv_transpose2d` remains unbound because the Stable Diffusion MLX example
  path uses nearest upsample plus convolution rather than transposed
  convolution.
