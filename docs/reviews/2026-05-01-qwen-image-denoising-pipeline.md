# Qwen-Image Denoising Pipeline Review

## Summary

Qwen-Image now has a prepared-embedding denoising loop that owns the base
FlowMatch sampling contract without crossing into prompt encoding or edit
variants.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Scope

This tranche wires the prepared-embedding Qwen-Image sampling path. It adds
packed-latent denoising over `FlowMatchEulerScheduler`, external true-CFG
composition, and a generation helper that samples packed latents before VAE
decode. Prompt encoding, tokenizer orchestration, edit/inpaint/control/layered
variants, guidance-distilled variants, and image-file output remain out of
scope.

## Reference Parity

The local loop follows Diffusers `QwenImagePipeline` behavior for the base
text-to-image path: packed latents are `[B, imageSeqLen, latentChannels * 4]`,
the transformer receives `timestep / 1000`, scheduler stepping uses the
original FlowMatch timestep, and true CFG is computed as a second negative
denoiser pass followed by norm rescaling to the conditional prediction norm.

Local `@mlxts/diffusion` keeps prompt encoding outside this package so
`@mlxts/diffusion` does not import autoregressive transformer surfaces.

## Tensor Lifetime Audit

`denoiseQwenImageLatents` retains the caller's initial latents, replaces the
current latent after each scheduler step, and frees the retained current latent
on failure. Denoiser predictions, scaled latent inputs, timestep tensors, CFG
intermediates, latent unpacking, VAE statistic tensors, and decode temporaries
use `using` declarations. Returned tensors are owned by the caller.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/families/qwen-image`

The loop adds no performance claim. The base path performs one transformer
prediction per denoising step. True CFG intentionally performs two predictions
per step, matching the Diffusers contract for classifier-free guidance.

## Independent Review

Dirac the 2nd reviewed the local Qwen-Image files and Diffusers references
before implementation. The recommendation was to land only a prepared-embedding
denoising helper, enforce packed-latent and prompt tensor shapes, pass
normalized timesteps to the transformer, keep CFG outside the transformer, and
leave prompt encoding plus edit/control variants out of scope.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Full Qwen-Image product generation still needs prompt/tokenizer/Qwen2.5-VL
conditioning, real checkpoint image proof, and example-level image output.
