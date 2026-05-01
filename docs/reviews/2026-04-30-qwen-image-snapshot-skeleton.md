# Qwen-Image Snapshot Skeleton Review

## Summary

Base Diffusers `QwenImagePipeline` snapshots are now recognized by
`@mlxts/diffusion`, Qwen-Image transformer/VAE configs parse into package-owned
metadata, and FlowMatch Euler accepts the Qwen-used `shift_terminal` field.
Runtime tensor execution remains outside this tranche.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/config.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/flow-match-scheduler-config.ts`
- `packages/diffusion/src/schedulers/flow-match-euler.ts`
- `packages/diffusion/src/index.ts`

## Scope

This tranche adds Qwen-Image snapshot recognition and package-native config
parsing only. It does not add Qwen-Image transformer, text-encoder, VAE tensor
execution, or image generation claims.

## Reference Evidence

- Diffusers `QwenImagePipeline` composes `QwenImageTransformer2DModel`,
  `AutoencoderKLQwenImage`, `FlowMatchEulerDiscreteScheduler`,
  `Qwen2_5_VLForConditionalGeneration`, and a Qwen2 tokenizer.
- Diffusers `QwenImageTransformer2DModel` packs 2x2 latent patches, defaults to
  64 input channels, 16 output latent channels, 60 layers, 24 heads, 128 head
  dim, and RoPE axes `[16, 56, 56]`.
- Diffusers `AutoencoderKLQwenImage` is a Qwen-specific 3D causal VAE derived
  from Wan-style video VAE structure, with `z_dim=16`, `base_dim=96`,
  `dim_mult=[1,2,4,4]`, and `temperal_downsample=[false,true,true]`.
- Diffusers FlowMatch Euler supports `shift_terminal`; current Qwen-Image
  snapshots use it.

## Tensor Lifetime Audit

The Qwen-Image skeleton adds JSON parsing and scheduler scalar schedule math.
It does not introduce new `MxArray` construction, disposal, nested tensor
expressions, native handles, or MLX eval points.

## Memory / Performance Evidence

No model hot path is added in this tranche. The only runtime behavior change is
scalar FlowMatch sigma schedule stretching when `shiftTerminal` is present.

Focused validation before full gates:

- `bun test packages/diffusion/src/families/qwen-image packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/pretrained/scheduler-config.test.ts packages/diffusion/src/schedulers/flow-match-euler.test.ts`
- `bun run --filter '@mlxts/diffusion' typecheck`

Both passed locally before this review artifact was written.

## Independent Review

Ampere (`019de0f5-872a-7c93-904a-c5a8f50788bb`) reviewed the Qwen-Image
snapshot skeleton plan against Diffusers and cached snapshot metadata. The
review called out the Qwen-specific VAE boundary, the `shift_terminal` scheduler
field, and keeping Qwen text encoding as manifest metadata rather than importing
`@mlxts/transformers` into `@mlxts/diffusion`.

## Remaining Risks / Follow-ups

- Real Qwen-Image checkpoint execution remains unimplemented.
- Qwen-Image edit, ControlNet, inpaint, img2img, and layered pipeline variants
  remain intentionally unsupported until their runtime semantics are designed.
- A real cached base checkpoint proof is still needed once model execution
  exists.
- Text conditioning will need an explicit boundary for Qwen2.5-VL embeddings
  without making `@mlxts/diffusion` import autoregressive model families.
