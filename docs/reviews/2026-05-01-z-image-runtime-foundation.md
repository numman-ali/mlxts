# Runtime Review: Z-Image Runtime Foundation

## Summary

Implemented the package-owned base Z-Image text-to-image tensor path: latent patching, Z-specific RoPE, single-stream attention, modulated/unmodulated transformer blocks, denoising over prepared caption embeddings, and Diffusers transformer weight mapping/loading. The runtime is intentionally scoped to dense Diffusers Z-Image snapshots, batch size 1, and no-CFG text-to-image semantics.

## Files Reviewed

- `packages/diffusion/src/families/z-image/attention.ts`
- `packages/diffusion/src/families/z-image/blocks.ts`
- `packages/diffusion/src/families/z-image/embeddings.ts`
- `packages/diffusion/src/families/z-image/latents.ts`
- `packages/diffusion/src/families/z-image/pipeline.ts`
- `packages/diffusion/src/families/z-image/tensor-utils.ts`
- `packages/diffusion/src/families/z-image/transformer.ts`
- `packages/diffusion/src/families/z-image/weight-mapping.ts`
- `packages/diffusion/src/families/z-image/weights.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

All tensor-producing helper calls in the new runtime path keep intermediates visible through `using` declarations or explicit free paths. Variable-length helper objects (`ZImagePaddedFeature`, `PreparedSequence`, denoiser input latents) have paired disposal in `finally` blocks. The transformer path retains only returned tensors and frees patchified, padded, prepared, and block-loop intermediates before exit. `bun run check:tensor-lifetimes` passed after implementation.

## Memory / Performance Evidence

Focused fixture tests cover patchify/unpatchify round-trip, sequence padding, base transformer forward shape, no-CFG denoising, decode layout conversion, and weight-name routing. No real checkpoint performance claim is made in this tranche. The implementation preserves the Diffusers runtime shape for the base path: `[C,F,H,W]` internal latent samples, sequence padding to 32, learned pad tokens, normalized timestep `(1000 - t) / 1000`, and negated transformer output before the FlowMatch Euler step.

## Independent Review

Arendt (`019de17b-93f4-75e2-91c6-611f4e2950d6`) performed a read-only architecture spike against the local Diffusers Z-Image reference and recommended the same base-text-to-image-only tranche boundary. The review called out FLUX reuse limits, the dense Diffusers weight map, and the current official snapshot availability blocker.

## Remaining Risks / Follow-ups

The official `Tongyi-MAI/Z-Image-Turbo` Diffusers snapshot is not yet present in the local Hugging Face cache, so the real checkpoint proof remains pending. The cached `filipstrand/Z-Image-Turbo-mflux-4bit` snapshot uses mflux quantized sidecars and lacks the Diffusers metadata contract; it is out of scope for this dense-runtime tranche. Omni/SigLIP, ControlNet, img2img/inpaint, multi-batch, CFG, and mflux quant support remain explicit future tranches.
