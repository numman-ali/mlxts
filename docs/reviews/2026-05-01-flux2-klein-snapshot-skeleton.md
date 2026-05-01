# FLUX.2 Klein Snapshot Skeleton

## Summary

Added parse-only support for current Diffusers `Flux2KleinPipeline` snapshots.
`@mlxts/diffusion` now recognizes the model index, records `is_distilled`, and
parses `Flux2Transformer2DModel` plus `AutoencoderKLFlux2` component configs as
a separate family contract. This does not add tensor execution, weight loading,
prompt encoding, image/reference conditioning, or an example proof command.

References checked:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- https://huggingface.co/docs/diffusers/api/pipelines/flux2
- `.reference/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_flux2.py`
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`

## Files Reviewed

- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/families/flux2/config.ts`
- `packages/diffusion/src/families/flux2/config-parsing.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

This tranche changes snapshot and JSON config parsing only. It constructs no
`MxArray`, loads no safetensors, and executes no denoising, VAE, transformer, or
scheduler tensor path.

## Memory / Performance Evidence

No runtime tensor path changed, so no generation or image benchmark claim is
made. The added tests prove metadata parsing, scheduler manifest acceptance, and
VAE packed-latent-channel agreement with transformer input channels.

## Independent Review

Newton performed a read-only reference pass and recommended a separate `flux2`
family skeleton with only `Flux2KleinPipeline`, `Flux2Transformer2DModel`,
`AutoencoderKLFlux2`, `Qwen3ForCausalLM`, and `Qwen2TokenizerFast` support.
Newton also called out the VAE packed-latent-channel agreement check, which is
now enforced in `loadFlux2KleinComponentConfigs`.

## Remaining Risks / Follow-ups

- Runtime tensor execution, weight loading, prompt encoding, Qwen3 text
  execution, tokenizer execution, VAE decode, examples, and proof commands are
  not implemented.
- `Flux2Pipeline`, `Flux2KleinInpaintPipeline`, `Flux2KleinKVPipeline`,
  modular Flux2 pipelines, single-file/GGUF/quant sidecars, LoRA, KV-cache
  behavior, inpainting, and image/reference conditioning remain explicitly
  unsupported.
- Larger product characterization waits until the runtime path exists.
