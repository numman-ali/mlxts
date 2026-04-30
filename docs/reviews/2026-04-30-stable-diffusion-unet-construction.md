# Stable Diffusion UNet Construction

## Summary

Added package-owned Stable Diffusion `UNet2DConditionModel` construction for
`@mlxts/diffusion`. The tranche implements NHWC latent flow, Diffusers-style
timestep embeddings, down/mid/up residual ordering, spatial transformer
self/cross attention, SDXL text-time conditioning, `use_linear_projection`
projection shape semantics, and focused tests for shape, parameter tree, and
construction guardrails.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/config.ts`
- `packages/diffusion/src/families/stable-diffusion/unet.ts`
- `packages/diffusion/src/families/stable-diffusion/unet-blocks.ts`
- `packages/diffusion/src/families/stable-diffusion/unet-embeddings.ts`
- `packages/diffusion/src/families/stable-diffusion/unet-transformer.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/models/unets/unet_2d_condition.py`
  confirms top-level UNet ordering, timestep projection, SDXL text-time
  conditioning, reversed channel schedules for up blocks, and final
  normalization/activation/convolution.
- `.reference/diffusers/src/diffusers/models/unets/unet_2d_blocks.py`
  confirms down/up block residual collection, one residual pop per up resnet,
  cross-attention block placement, stride-2 UNet downsampling, and nearest
  upsample plus convolution.
- `.reference/diffusers/src/diffusers/models/embeddings.py` confirms the
  sinusoidal timestep formula: inverse-frequency denominator
  `half_dim - downscale_freq_shift`, sine/cosine concatenation, optional
  `flip_sin_to_cos`, and odd-dimension zero padding.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/unet.py` confirms
  MLX NHWC flow, model-stage ordering, residual stack structure, and
  `text_time` conditioning shape.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  highlights later weight-loading pitfalls: 1x1 transformer projections are
  convolutional when `use_linear_projection=false`, and Diffusers GEGLU uses a
  fused projection tensor.

## Tensor Lifetime Audit

The forward path keeps tensor ownership local and visible. Down blocks return
owned hidden states plus retained residual snapshots; the top-level UNet frees
the previous hidden state after each stage and frees all unconsumed residuals in
a `finally` block. Up blocks pop residuals from the shared stack, consume each
residual through lexical `using`, and return one owned hidden tensor. Transformer
and attention intermediates use lexical `using`; split GEGLU tensors are freed
explicitly in `finally`.

## Memory / Performance Evidence

No throughput benchmark is claimed for this construction-only tranche. The
implementation preserves the package's channel-last MLX layout and uses fused
`scaledDotProductAttention` for transformer self/cross attention. It does not
introduce image generation, checkpoint loading, or serving behavior.

Focused validation passed:

- `bun test packages/diffusion/src/families/stable-diffusion/unet.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion`
- `bun run --filter @mlxts/diffusion typecheck`

## Independent Review

Mencius completed a read-only second pass over the package code and local
Diffusers/MLX references. The review corrected two construction details before
landing: `Transformer2D` projections honor `useLinearProjection`, and GEGLU
uses a fused input projection so future Diffusers weight loading does not need
an avoidable split-weight path. The review also confirmed the residual stack
ordering, NHWC tensor layout, UNet downsample semantics, SDXL text-time
conditioning contract, and package boundary.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

- UNet safetensor mapping/loading remains the next Stable Diffusion checkpoint
  tranche.
- Full image-generation parity still requires text encoder/tokenizer boundary
  work, scheduler orchestration, classifier-free guidance, latent scaling, and
  VAE decode proof.
