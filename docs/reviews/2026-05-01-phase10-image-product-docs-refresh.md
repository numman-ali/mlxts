# Phase 10 Image Product Docs Refresh

## Summary

Refreshed the top-level product docs so the repo advertises the current Phase
10 image-generation state honestly. Stable Diffusion / SDXL, FLUX.1,
Z-Image-Turbo, and Qwen-Image / Qwen-Image-2512 are the implemented
finite-proof ladder with bounded real checkpoint evidence. FLUX.2 Klein,
Stable Diffusion 3 / 3.5, image editing modes, video, and audio remain separate
later tranches.

## Files Reviewed

- `README.md`
- `docs/ecosystem-structure.md`
- `PLAN.md`
- `packages/diffusion/README.md`
- `docs/reviews/2026-05-01-phase10-image-model-roadmap.md`

## Reference Check

- Diffusers exposes Z-Image through `ZImagePipeline` and its image-to-image and
  inpainting variants as distinct pipeline surfaces. Source:
  https://huggingface.co/docs/diffusers/en/api/pipelines/z_image
- `Tongyi-MAI/Z-Image-Turbo` remains the right speed-first Z-Image target for
  local proof work because it is the public Diffusers Z-Image-Turbo checkpoint.
  Source: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Diffusers exposes Qwen-Image through `QwenImagePipeline` with
  `AutoencoderKLQwenImage`, Qwen2.5-VL text encoding, FlowMatch Euler, and
  separate img2img, inpaint, and edit variants. Source:
  https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/qwenimage
- `Qwen/Qwen-Image-2512` remains the forward Qwen image target in the current
  model catalog. Source: https://huggingface.co/Qwen/Qwen-Image-2512
- FLUX.2 Klein is a separate family surface: the model card uses
  `Flux2KleinPipeline`, and the official BFL docs describe the open-weight
  Klein models separately from FLUX.1. Sources:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B and
  https://docs.bfl.ai/flux_2/flux2_overview
- Stable Diffusion 3 / 3.5 remains a separate MMDiT/flow target because the
  Diffusers pipeline uses `SD3Transformer2DModel`, FlowMatch Euler, an
  AutoencoderKL, two CLIP encoders, and a T5 encoder. Sources:
  https://huggingface.co/docs/diffusers/v0.31.0/en/api/pipelines/stable_diffusion/stable_diffusion_3
  and https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- SDXL remains the Stable Diffusion baseline because Diffusers documents it as
  the base text-to-image pipeline with two text encoders plus SDXL-specific
  conditioning. Source:
  https://huggingface.co/docs/diffusers/v0.22.0/using-diffusers/sdxl

## Outcome

`README.md` now lists the Phase 10 image proof workbooks in the repo shape,
shows the four current finite proof commands, and moves stale Stable
Diffusion-only proof deferral into the correct future-work bucket:
quality/performance characterization for proved families plus later families
and modes.

`docs/ecosystem-structure.md` now states that SDXL, FLUX.1, Z-Image-Turbo, and
Qwen-Image-2512 runtime/proof paths are in place with bounded real checkpoint
evidence.

## Validation

- Documentation-only change.

## Remaining Risks / Follow-ups

- Larger/default-step image quality and performance characterization remains
  future work for every proved image family.
- FLUX.2 Klein and Stable Diffusion 3 / 3.5 need separate reference spikes
  before implementation because their pipeline/component shapes are not FLUX.1
  or SDXL variants.
