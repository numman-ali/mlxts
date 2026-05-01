# Phase 10 Image Model Roadmap

## Summary

Refreshed the Phase 10 image-generation support order against current
Hugging Face and Diffusers references. The roadmap now keeps SD/SDXL and
FLUX.1 as the near-term proof foundation, moves Z-Image-Turbo ahead of full
Qwen-Image runtime because it is the cleaner speed-first runtime target, names
`Qwen/Qwen-Image-2512` as the forward Qwen image target, and treats FLUX.2
Klein as a separate later family rather than a FLUX.1 variant.

## Files Reviewed

- `PLAN.md`
- `docs/gates-and-milestones.md`
- `docs/ecosystem-structure.md`
- `continuity.md`
- `MEMORY.md`

## Reference Audit

- Stable Diffusion XL remains the baseline because Diffusers exposes
  `StableDiffusionXLPipeline` over AutoencoderKL, two CLIP text encoders,
  UNet2DConditionModel, scheduler metadata, and SDXL-specific conditioning
  fields. Source: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
- `black-forest-labs/FLUX.1-schnell` remains the first flow target after
  SD/SDXL because it is a Diffusers `FluxPipeline` checkpoint with Apache-2.0
  licensing and an already-landed package path. Source:
  https://huggingface.co/black-forest-labs/FLUX.1-schnell
- Z-Image is current in Diffusers as `ZImagePipeline` over FlowMatch Euler,
  AutoencoderKL, Qwen-compatible text encoding, and `ZImageTransformer2DModel`.
  `Tongyi-MAI/Z-Image-Turbo` is the speed-first target because its model card
  positions Turbo as an 8-NFE distilled generation model that fits the local
  product goal. Sources:
  https://huggingface.co/docs/diffusers/api/pipelines/z_image and
  https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Qwen-Image remains architecturally important but heavier. Diffusers exposes
  `QwenImagePipeline`, while `Qwen/Qwen-Image-2512` is the newer forward target
  with improved realism, detail, and text rendering. Its Qwen-specific VAE and
  text stack make runtime execution a separate tranche after the existing
  config/model-index skeleton. Sources:
  https://huggingface.co/docs/diffusers/api/pipelines/qwenimage and
  https://huggingface.co/Qwen/Qwen-Image-2512
- FLUX.2 Klein is not a FLUX.1 checkpoint variant. Diffusers exposes
  `Flux2KleinPipeline`, and the model card describes `black-forest-labs/FLUX.2-klein-4B`
  as a 4B model with a separate reference implementation and Diffusers support.
  Sources: https://huggingface.co/docs/diffusers/en/api/pipelines/flux2 and
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Stable Diffusion 3 / 3.5 remains later because it uses MMDiT and three text
  encoders, and the public SD3.5 Large checkpoint is access-gated under the
  Stability community license. Source:
  https://huggingface.co/stabilityai/stable-diffusion-3.5-large

## Outcome

The support ladder is:

1. Stable Diffusion / SDXL baseline.
2. FLUX.1.
3. Z-Image-Turbo.
4. Qwen-Image / Qwen-Image-2512.
5. FLUX.2 Klein.
6. Stable Diffusion 3 / 3.5 and compatible distilled variants.

This is a product support order, not a claim that every listed family has
runtime tensor execution today.

## Validation

- Documentation-only change.
- `bun run validate`

## Remaining Risks / Follow-ups

- FLUX.1 still needs real checkpoint image evidence before it can be called
  product-complete.
- Z-Image and Qwen-Image runtime execution remain separate implementation
  tranches.
- Video and audio generation remain Phase 10 work after the first image
  generation surfaces are stable.
