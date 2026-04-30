# Stable Diffusion Pipeline Assembly

## Summary

Added a package-owned Stable Diffusion sampling boundary over supplied
conditioning tensors. The tranche keeps text encoders and tokenizers outside
`@mlxts/diffusion`, while `@mlxts/diffusion` owns NHWC latent shape selection,
initial noise sampling, DDIM/Euler denoising steps, classifier-free guidance,
VAE latent unscale, and 0..1 image postprocessing.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/autoencoder.ts`
- `packages/diffusion/src/families/stable-diffusion/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/mlx-examples/stable_diffusion/stable_diffusion/__init__.py`
  confirms the denoising loop shape: condition externally, sample initial
  latents, concatenate negative and positive branches for CFG, step the sampler,
  and decode through the VAE after denoising.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/sampler.py`
  confirms the Euler sigma scaling and step equations used by the existing
  scheduler implementation.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  confirms negative-first CFG ordering and `latents / vae.scaling_factor`
  before VAE decode.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py`
  confirms the same single-UNet-call CFG batching pattern.

## Tensor Lifetime Audit

The sampling loop retains the caller-provided initial latents before mutation,
then frees each previous latent after the next step is created. Scheduler-scaled
latents, guided batch latents, concatenated conditioning, UNet predictions, DDIM
`predOriginalSample`, and VAE decode intermediates are scoped with `using` or
explicit `free()` paths. The loop evaluates each denoising step by default so a
multi-step sample does not accumulate one large lazy graph.

## Memory / Performance Evidence

This tranche adds a new pipeline path and does not claim generation throughput
improvement. It avoids text-encoder imports and keeps host image encoding out of
the runtime path. The default per-step `mxEval()` is a deliberate memory posture
for long denoising loops; future performance work can introduce a measured
batched-eval or compiled denoising strategy behind the same semantic surface.

## Independent Review

Aquinas reviewed the planned Phase 10 diffusion pipeline tranche before final
edits. The review recommended supplied tensor conditioning, negative-first CFG
batching, NHWC latent layout, explicit scheduler adaptation, VAE scaling outside
the VAE module, and per-step evaluation to prevent graph growth. Those points
are reflected in the landed implementation.

## Validation

- `bun test packages/diffusion/src/families/stable-diffusion/pipeline.test.ts packages/diffusion/src/families/stable-diffusion/autoencoder.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:cross-package-imports`
- `bun run check:skills`
- `bun run validate`

## Remaining Risks / Follow-ups

- The pipeline currently consumes already-expanded conditioning tensors. CLIP
  tokenization/text encoding and SDXL prompt embedding composition remain a
  separate `@mlxts/transformers` or example composition tranche.
- Real checkpoint image proof still needs a conditioning source, host image
  encoding, and an AXI-shaped finite proof command before Phase 10 diffusion is
  complete.
