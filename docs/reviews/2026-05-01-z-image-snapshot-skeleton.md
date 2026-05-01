# Z-Image Snapshot Skeleton Review

## Summary

Base Diffusers `ZImagePipeline` snapshots are now recognized by
`@mlxts/diffusion`, and Z-Image transformer plus VAE configs parse into
package-owned metadata. Runtime tensor execution and prompt encoding remain
outside this tranche.

## Files Reviewed

- `packages/diffusion/src/families/z-image/config.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/index.ts`

## Scope

This tranche adds Z-Image snapshot recognition and package-native config
parsing only. It does not add Z-Image transformer execution, Qwen3 text
encoding, VAE tensor execution, or image generation claims.

## Reference Evidence

- Current HF `Tongyi-MAI/Z-Image-Turbo` `model_index.json` exposes
  `ZImagePipeline` over `ZImageTransformer2DModel`, standard `AutoencoderKL`,
  `FlowMatchEulerDiscreteScheduler`, `Qwen3Model`, and `Qwen2Tokenizer`.
- Diffusers `ZImageTransformer2DModel` registers `all_patch_size`,
  `all_f_patch_size`, `in_channels`, `dim`, `n_layers`, `n_refiner_layers`,
  `n_heads`, `n_kv_heads`, `norm_eps`, `qk_norm`, `cap_feat_dim`,
  `siglip_feat_dim`, `rope_theta`, `t_scale`, `axes_dims`, and `axes_lens`.
- Diffusers asserts `dim / n_heads === sum(axes_dims)` and uses
  `SEQ_MULTI_OF=32` plus `X_PAD_DIM=64` in the patch/pad path.
- The base Z-Image pipeline uses Qwen chat-template prompt encoding, takes
  `hidden_states[-2]`, and passes masked variable-length embeddings. That
  stays an application or transformer-side conditioning concern.
- Diffusers has separate Z-Image ControlNet, inpaint, img2img, and Omni
  pipelines; the first tranche recognizes only base `ZImagePipeline`.

## Tensor Lifetime Audit

The Z-Image skeleton adds JSON parsing only. It does not introduce new
`MxArray` construction, disposal, nested tensor expressions, native handles, or
MLX eval points.

## Memory / Performance Evidence

No model hot path is added in this tranche. Focused validation before full
gates:

- `bun test packages/diffusion/src/families/z-image packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/pretrained/scheduler-config.test.ts`
- `bun run --filter '@mlxts/diffusion' typecheck`
- `bun run check:file-lines`

All passed locally before this review artifact was written.

## Independent Review

Mill (`019de106-6873-7fc1-b05d-c1184b28b294`) completed a read-only second pass
against Diffusers and local roadmap docs. The review recommended a bounded
Z-Image snapshot/config skeleton, confirmed the component tuple, called out the
Qwen chat-template prompt boundary, and warned against accepting FLUX.2 or SD3
as widened variants of existing family surfaces.

## Remaining Risks / Follow-ups

- Real Z-Image checkpoint execution remains unimplemented.
- Qwen3 text conditioning for Z-Image needs an explicit application or
  transformer-side boundary without importing `@mlxts/transformers` into
  `@mlxts/diffusion`.
- Z-Image ControlNet, inpaint, img2img, and Omni variants remain intentionally
  unsupported until their runtime semantics are designed.
- The cached `filipstrand/Z-Image-Turbo-mflux-4bit` snapshot has mflux-style
  folders without Diffusers `model_index.json` and component configs; supporting
  that layout is a separate import/metadata tranche.
