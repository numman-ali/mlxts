# Phase 10 Image Model Roadmap

## Summary

Refreshed the Phase 10 image-generation support order against current
Hugging Face and Diffusers references. SD/SDXL, FLUX.1, Z-Image-Turbo, and
Qwen-Image / Qwen-Image-2512 and FLUX.2 Klein now have bounded real checkpoint
image evidence. FLUX.2 Klein remains a separate later family rather than a
FLUX.1 variant. Stable Diffusion 3 / 3.5 now has snapshot/config skeleton
support as a separate MMDiT/flow target.

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
  text stack now have a package-owned runtime/proof-command path, and official
  bounded checkpoint evidence has landed. Sources:
  https://huggingface.co/docs/diffusers/api/pipelines/qwenimage and
  https://huggingface.co/Qwen/Qwen-Image-2512
- FLUX.2 Klein is not a FLUX.1 checkpoint variant. Diffusers exposes
  `Flux2KleinPipeline`, and the model card describes `black-forest-labs/FLUX.2-klein-4B`
  as a 4B model with a separate reference implementation and Diffusers support.
  Sources: https://huggingface.co/docs/diffusers/en/api/pipelines/flux2 and
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Stable Diffusion 3 / 3.5 uses Diffusers `StableDiffusion3Pipeline` over
  `SD3Transformer2DModel`, FlowMatch Euler, AutoencoderKL, two CLIP projection
  encoders, and a T5 encoder. The public SD3.5 Large checkpoint remains
  access-gated under the Stability community license, so real proof needs an
  authenticated follow-up. Sources:
  https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3 and
  https://huggingface.co/stabilityai/stable-diffusion-3.5-large

## Outcome

The support ladder is:

1. Stable Diffusion / SDXL baseline.
2. FLUX.1.
3. Z-Image-Turbo.
4. Qwen-Image / Qwen-Image-2512.
5. FLUX.2 Klein.
6. Stable Diffusion 3 / 3.5 and compatible distilled variants.

Stable Diffusion / SDXL, FLUX.1, Z-Image-Turbo, Qwen-Image / Qwen-Image-2512,
and FLUX.2 Klein all have bounded real checkpoint evidence. Stable Diffusion 3
/ 3.5 has snapshot/config recognition but no package-owned runtime or real
checkpoint proof yet.

## Tensor Lifetime Audit

This roadmap artifact changes documentation only. Runtime tensor ownership is
audited in the paired real-checkpoint proof artifacts for the implemented image
families.

## Memory / Performance Evidence

This roadmap artifact carries no performance claim. Bounded checkpoint evidence
for implemented image families is recorded in their dedicated runtime review
artifacts, and larger/default-step characterization remains follow-up work.

## Independent Review

Hooke performed a read-only product/architecture pass across Phase 10 image
support and recommended proving Z-Image-Turbo and Qwen-Image-2512 before
advancing later image families. Galileo later reviewed the Qwen2 tokenizer
boundary needed by the Qwen-Image proof.

## Validation

- Documentation-only change.
- `bun run validate`

## Remaining Risks / Follow-ups

- Z-Image-Turbo has bounded official dense checkpoint image evidence, but still
  needs larger/default-step quality and performance characterization before it
  can be called product-complete.
- Qwen-Image / Qwen-Image-2512 has bounded official checkpoint image evidence,
  but still needs larger/default-step quality and performance characterization
  before it can be called product-complete.
- FLUX.2 Klein has bounded official checkpoint image evidence, but still needs
  reference-image/KV variants plus larger/default-step quality and performance
  characterization before it can be called product-complete.
- Stable Diffusion 3 / 3.5 has manifest/config skeleton support only; runtime
  tensor execution, weight mapping/loading, prompt encoding, and authenticated
  official checkpoint proof remain follow-up work.
- Image-to-image, inpainting, ControlNet, edit/control/layered variants, video,
  and audio generation remain Phase 10 work after the first text-to-image
  surfaces are stable.
