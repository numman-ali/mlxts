# FLUX Pipeline Contract

## Summary

Added the tensor-only FLUX sampling contract over prepared conditioning tensors.
The tranche wires packed latent sampling, FlowMatch Euler denoising, FLUX text
and image ids, embedded guidance passthrough, and FLUX VAE latent normalization.

This does not add the FLUX transformer module, CLIP/T5 composition, checkpoint
weight loading, or a full text-to-image CLI.

## Files Reviewed

- `packages/diffusion/src/families/flux/config.ts`
- `packages/diffusion/src/families/flux/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py`
  prepares packed latent image sequences, unbatched image ids, FlowMatch
  timesteps keyed by image sequence length, and VAE decode with
  `(latents / scaling_factor) + shift_factor`.
- `.reference/mlx-examples/flux/flux/flux.py` confirms the MLX Flux loop calls
  the transformer with packed `img`, `img_ids`, prepared T5 `txt`, text ids,
  CLIP pooled `y`, rank-1 timesteps, and optional embedded guidance.
- `.reference/mlx-examples/flux/flux/utils.py` confirms FLUX.1 VAE latent
  channels 16, scaling factor 0.3611, shift factor 0.1159, and 8x VAE scale.
- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py` keeps
  FLUX guidance as a transformer input rather than Stable Diffusion-style
  negative-prompt classifier-free batch duplication.

## Tensor Lifetime Audit

`createFluxInitialLatents()` names the sampled NHWC latent tensor with a
`using` binding before returning the packed latent tensor. `denoiseFluxLatents()`
retains caller-owned initial latents, image ids, and text ids when needed, frees
owned ids in `finally`, and frees the current latent on error. Per-step
temporaries for scaled latents, timestep tensors, and model predictions stay
visible through `using` bindings.

`decodeFluxLatents()` names unpack, scale, shift, decode, normalize, and clamp
intermediates before returning the final image tensor.

`config.ts` parses JSON metadata only and does not allocate tensors.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/families/flux`: 19 pass, 0 fail.
- `bun run --filter @mlxts/diffusion typecheck`: passed.
- `bun run check:tensor-lifetimes`: passed.
- `bun run check:runtime-review`: passed and validated this artifact.
- `bun run check:coverage`: passed; `@mlxts/diffusion` coverage is 96.03%
  lines and 94.95% functions.
- `bun run validate`: passed.

The tranche adds the FLUX sampling control path but no transformer hot path and
no checkpoint-backed generation proof. It makes no image quality or throughput
claim.

## Independent Review

McClintock completed a read-only second pass before implementation. The review
recommended landing the FLUX tensor-only sampling seam before the transformer
backbone, keeping CLIP/T5 composition outside `@mlxts/diffusion`, preserving
FLUX VAE shift semantics, and treating Qwen-Image and Z-Image as future
contracts rather than FLUX aliases.

## Remaining Risks / Follow-ups

- FLUX transformer blocks, RoPE application, guidance embedding, CLIP plus T5
  conditioning, VAE module loading, checkpoint weight mapping, and real image
  proof remain follow-on Phase 10 tranches.
- Qwen-Image uses a different VAE normalization contract with mean/std and
  rank-5 latent handling. It must not reuse this FLUX decoder path blindly.
- Z-Image shares flow-style generation at the product level, but it needs its
  own reference audit before entering this package.

## Out-of-scope drift noticed

No out-of-scope drift was changed in this tranche.
