# Runtime Review: Stable Diffusion 3 Runtime Foundation

## Summary

Implemented the package-owned Stable Diffusion 3 / 3.5 prepared-tensor runtime foundation. The tranche covers NHWC latent patch embedding, fixed 2D sincos positional crops, SD3 MMDiT joint attention, SD3.5 RMS q/k norm and dual-attention blocks, FlowMatch Euler denoising over prepared conditioning tensors, classifier-free guidance batching, and the VAE shift/scale decode boundary.

No weight loading, prompt encoding, image proof, IP-Adapter, ControlNet, skip-layer guidance, img2img, inpaint, PAG, LoRA, or generation quality claim is made in this tranche.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion-3/attention.ts`
- `packages/diffusion/src/families/stable-diffusion-3/blocks.ts`
- `packages/diffusion/src/families/stable-diffusion-3/embeddings.ts`
- `packages/diffusion/src/families/stable-diffusion-3/latents.ts`
- `packages/diffusion/src/families/stable-diffusion-3/normalization.ts`
- `packages/diffusion/src/families/stable-diffusion-3/pipeline.ts`
- `packages/diffusion/src/families/stable-diffusion-3/tensor-utils.ts`
- `packages/diffusion/src/families/stable-diffusion-3/transformer.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- Hugging Face Diffusers documents `StableDiffusion3Pipeline` as `SD3Transformer2DModel`, `FlowMatchEulerDiscreteScheduler`, `AutoencoderKL`, two CLIP projection encoders, and one T5 encoder:
  https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3
- Local Diffusers `SD3Transformer2DModel` source uses `PatchEmbed`, `CombinedTimestepTextProjEmbeddings`, `JointTransformerBlock`, `AdaLayerNormContinuous`, and final unpatchify:
  `.reference/diffusers/src/diffusers/models/transformers/transformer_sd3.py`
- Local Diffusers `JointTransformerBlock` source confirms image/context attention, final `context_pre_only`, SD3.5 `SD35AdaLayerNormZeroX`, optional `qk_norm`, and image-only dual attention:
  `.reference/diffusers/src/diffusers/models/attention.py`
- Local Diffusers attention processor confirms joint attention concatenates `[image, context]`, splits at image length, and only projects context output when `context_pre_only` is false:
  `.reference/diffusers/src/diffusers/models/attention_processor.py`

## Tensor Lifetime Audit

Tensor-producing calls in the new runtime path keep intermediates visible through `using` declarations or explicit `finally` disposal. The joint attention path disposes image/context projections after output projection, block loops free replaced image/context streams, cached positional embeddings are private non-parameter tensors disposed by `StableDiffusion3PatchEmbed`, and CFG-owned concatenated conditioning tensors have paired disposal.

`bun run check:tensor-lifetimes` passed after implementation.

## Memory / Performance Evidence

Focused synthetic tests cover patch embedding shape and crop rejection, timestep embedding finiteness, unpatchify layout, base SD3 transformer forward, SD3.5 q/k norm plus dual-attention parameter paths, final context-pre-only parameter omission, malformed tensor rejection, raw FlowMatch timestep values, CFG batching, and VAE shift/scale decode.

No real checkpoint performance claim is made. The implementation preserves the Diffusers runtime semantics needed before weight proof: NHWC package boundary over Diffusers NCHW meaning, raw FlowMatch timesteps, fixed 2D sincos position crops, `[image, context]` joint attention order, final context pre-only behavior, SD3.5 per-head RMS q/k norms, and SD3.5 image-only dual attention.

## Independent Review

Boole the 2nd (`019de3c5-27c1-7e42-8b6a-044ed0bc04a3`) performed a read-only SD3 runtime audit against local Diffusers references and local diffusion family patterns. The review recommended this exact tranche boundary: transformer plus thin prepared-tensor denoise surface, synthetic tests only, no weight loading or prompt encoding. It also called out the critical axis/order risks: NHWC package boundary, `[image, context]` attention order, fixed sincos position crop, raw timestep scale, `captionProjectionDim === hiddenSize`, RMS q/k norm placement, and SD3.5 9-way AdaLN-Zero-X.

## Remaining Risks / Follow-ups

Official Stability SD3/3.5 checkpoints are gated, so authenticated weight loading and proof remain separate. Prompt encoding still belongs outside `@mlxts/diffusion` and must compose two CLIP projection encoders plus T5 embeddings before this prepared runtime can generate real images. Weight mapping/loading, LoRA, IP-Adapter, ControlNet, skip-layer guidance, img2img/inpaint/PAG variants, and fused/chunked optimizations remain future tranches.
