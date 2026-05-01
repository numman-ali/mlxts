# Runtime Review: LTX Video Denoising Runtime

## Summary

This tranche adds the classic LTX-Video packed-latent denoising loop for
prepared prompt embeddings and attention masks.

The loop stays below full checkpoint generation: it accepts already-packed
video latents, prepared T5-side conditioning tensors, and a package-owned
FlowMatch scheduler. It passes raw scheduler timesteps, latent video geometry,
and VAE-derived RoPE interpolation scale into an LTX denoiser, then applies the
Euler scheduler update over packed latents.

This does not add LTX transformer blocks, T5 prompt encoding, VAE execution,
latent denormalization, latent upsampling, artifact writing, LTX-2 audio/video
denoising, or proof commands.

## Files Reviewed

- `packages/diffusion/src/families/ltx/pipeline.ts`
- `packages/diffusion/src/families/ltx/pipeline.test.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

The denoising loop retains caller-owned initial latents, replaces the retained
loop state one step at a time, and frees the current retained handle on success
replacement or error. Classifier-free guidance concatenates duplicated latents,
negative-first prompt embeddings, and negative-first attention masks inside a
bounded `try/finally` block, then frees owned guided conditioning tensors after
the denoiser call.

Guidance output splits are freed after the guided prediction tensor is built,
matching the existing SD3 ownership pattern. Caller-owned conditioning tensors,
masks, scheduler, and denoiser are never freed by the loop.

No model module fields, safetensor ownership paths, native bindings, or FFI
symbol declarations changed.

## Reference Parity

The loop follows the current Diffusers classic LTX pipeline behavior:

- packed BCFHW video latent geometry comes from `ltx/latents.ts`
- dynamic FlowMatch shift uses unpatched `latentFrames * latentHeight * latentWidth`
- denoiser timesteps use raw scheduler timestep values, not FLUX.2-style
  normalization
- classifier-free guidance is one negative-first batched denoiser call
- prompt attention masks are part of the prepared-conditioning contract
- `ropeInterpolationScale` is
  `[vaeTemporalCompressionRatio / frameRate, vaeSpatialCompressionRatio, vaeSpatialCompressionRatio]`

Reference files:

- `.reference/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx.py`

Focused tests cover raw timestep values, unpatched video-length dynamic shift
under patched latent packing, default RoPE interpolation scale, negative-first
CFG batching, attention-mask concatenation, guided prediction math, malformed
shape rejection before denoiser calls, and cleanup on denoiser failure.

## Memory / Performance Evidence

No full model generation hot path changed, and no performance claim is made.
The new loop is synthetic prepared-tensor runtime scaffolding for future LTX
transformer and VAE tranches.

Focused gate passed:

```bash
bun test packages/diffusion/src/families/ltx
```

Follow-on gates run for this tranche:

```bash
bun run lint
bun run check:file-lines
bun run typecheck
bun run check:runtime-review
bun run validate
```

## Independent Review

Hume ran a read-only reference review before this tranche was finalized. The
review confirmed that classic LTX differs from FLUX.2 by using raw timesteps,
standard video-length dynamic FlowMatch shift, and one batched CFG denoiser
call. It also called out prompt attention masks as semantic inputs; the package
API now requires `promptAttentionMask`, and guided sampling requires matching
negative masks.

## Remaining Risks / Follow-ups

The next LTX tranches are transformer execution, VAE decode/denormalization,
latent upsampling, and finite `examples/ltx-video` proof wiring. LTX-2
audio/video denoising remains separate because it has two latent streams,
audio-duration geometry, modality-specific guidance, and cross-modality
coordination beyond this classic LTX-Video loop.
