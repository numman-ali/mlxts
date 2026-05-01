# FLUX.2 Klein Sampling Foundation

## Summary

Added the first prepared-embedding FLUX.2 Klein sampling foundation in
`@mlxts/diffusion`. The tranche covers NCHW 2x2 VAE latent patching,
Diffusers-style packed latent sequences, 4-axis image/text ids, empirical
FlowMatch dynamic shift, external classifier-free guidance for non-distilled
checkpoints, distilled-guidance suppression, and the VAE batch-norm inverse
decode boundary.

This is not a real checkpoint proof. It intentionally does not add
`Flux2Transformer2DModel` execution, Qwen3 prompt conditioning, VAE weight
loading, reference-image/KV variants, an example command, or performance
claims.

References checked:

- https://huggingface.co/docs/diffusers/api/pipelines/flux2
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://docs.bfl.ai/flux_2/flux2_overview
- `.reference/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_flux2.py`
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`

## Files Reviewed

- `packages/diffusion/src/families/flux2/latents.ts`
- `packages/diffusion/src/families/flux2/pipeline.ts`
- `packages/diffusion/src/families/flux2/decoding.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

Disposable `MxArray` intermediates are named with visible `using` lifetimes in
latent packing, denoising, and decode helpers. The denoising loop retains the
current latent state deliberately, frees the previous state after each scheduler
step, and frees retained id tensors in `finally`. Decode reverses VAE
batch-norm statistics through named intermediates before returning a retained
NHWC image tensor. Tests include a denoiser failure case that exercises cleanup
of retained denoising state.

## Memory / Performance Evidence

No real FLUX.2 checkpoint path or transformer hot path was added, so no
throughput or quality claim is made. Focused synthetic tests cover shape
contracts, latent patch order, id axes, initial latent sampling, empirical
`mu`, external CFG, distilled CFG suppression, denoiser failure cleanup, and
VAE batch-norm inverse decode semantics.

Commands run:

- `bun test packages/diffusion/src/families/flux2`
- `bun run --filter @mlxts/diffusion typecheck`

## Independent Review

Leibniz performed a read-only reference review with GPT-5.5 xhigh and
recommended this narrower prepared-embedding sampling tranche before any full
transformer/VAE/proof implementation. The review specifically called out
FLUX.2's NCHW patchified latents, 4-axis ids, Qwen3 hidden-state conditioning
boundary, external CFG, distilled behavior, empirical `mu`, and AutoencoderKL
FLUX.2 batch-norm normalization as the minimum safe next step.

## Remaining Risks / Follow-ups

- `Flux2Transformer2DModel` runtime execution and safetensor weight mapping are
  not implemented.
- `AutoencoderKLFlux2` module construction and weight loading are not
  implemented.
- Qwen3 conditioning remains application-layer future work, including hidden
  layers `(9, 18, 27)` and tokenizer/chat-template details.
- Reference-image, inpainting, KV-cached, GGUF/single-file, LoRA, and quantized
  FLUX.2 variants remain unsupported.
- No `examples/flux2` proof command or real checkpoint evidence exists yet.

## Out-of-scope Drift Noticed

- Current public Diffusers docs also expose `Flux2KleinKVPipeline` for
  reference-image KV reuse. This tranche intentionally stays on text-only
  prepared embeddings so the cache contract can be designed with transformer
  execution in view instead of guessed from the pipeline wrapper.
