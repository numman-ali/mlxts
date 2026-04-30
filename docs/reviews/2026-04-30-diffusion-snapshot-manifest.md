# Diffusion Snapshot Manifest

## Summary

Added local Diffusers snapshot inspection for `@mlxts/diffusion`. The package
now parses supported Stable Diffusion, Stable Diffusion XL, and Flux
`model_index.json` files, classifies component folders, validates local
component metadata/weight presence, and reuses scheduler-config parsing so
unsupported scheduler semantics still fail closed.

This tranche does not construct UNet, VAE, text encoder, tokenizer, safety
checker, or image output objects. It does not add Hugging Face Hub download,
Diffusers pipeline execution, or serving routes.

## Files Reviewed

- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/snapshot-manifest.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/pipeline_utils.py` defines
  `DiffusionPipeline.config_name = "model_index.json"` and stores components as
  pipeline config entries.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  names the Stable Diffusion components: VAE, text encoder, tokenizer, UNet,
  scheduler, optional safety checker, optional feature extractor, and optional
  image encoder.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`
  names the SDXL dual text encoder/tokenizer layout and optional image
  processor/encoder components.
- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py`
  names the Flux transformer, VAE, flow-matching scheduler, CLIP/T5 encoders,
  and tokenizer components.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  confirms the local folder/file conventions for Stable Diffusion UNet, VAE,
  text encoder, tokenizer, and scheduler metadata.
- `.reference/mlx-examples/flux/flux/utils.py` confirms Flux component folders
  for transformer, VAE, CLIP, T5, and tokenizer assets.

## Tensor Lifetime Audit

This tranche only reads JSON metadata and filesystem entries. It does not create
or dispose `MxArray` values and does not add tensor-producing operations.

## Memory / Performance Evidence

No generation benchmark is required. Snapshot inspection is host-side metadata
work and does not touch model construction, denoising loops, scheduling hot
paths, or serving routes.

Focused tests cover Stable Diffusion, SDXL, and Flux model-index parsing, disabled
optional components, local manifest inspection, required metadata/weight file
validation, unsupported pipeline/component rejection, unsupported scheduler
semantics, missing `model_index.json`, and malformed JSON.

## Independent Review

Copernicus was asked for a read-only second pass on the next diffusion boundary,
with focus on Diffusers `model_index.json`, Stable Diffusion component folders,
MLX examples conventions, and whether the next tranche should inspect manifests
or parse UNet/VAE configs.

## Remaining Risks / Follow-ups

- Flux is recognized at the manifest level only. Flow-matching scheduler
  construction remains unsupported and fails closed.
- ControlNet, LoRA-attached pipelines, single-file checkpoints, DDUF, and custom
  pipeline class tuples remain unsupported and fail closed.
- UNet/VAE/text-encoder config parsing and weight loading remain follow-on
  tranches.
- Safety checker and feature extractor components are recognized as metadata,
  not executed.
- Hugging Face Hub resolution belongs in the broader diffusion checkpoint-loading
  tranche.
