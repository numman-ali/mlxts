# FLUX Config And Latent Helpers

## Summary

Added FLUX.1 transformer config translation and package-owned latent helpers
for the Diffusers FLUX pipeline shape. The tranche is intentionally limited to
fail-closed config parsing, NHWC 2x2 latent packing, inverse unpacking, and
unbatched latent image position ids.

This does not add the FLUX transformer module, text encoder composition, VAE
integration, weight loading, or a full image sampling pipeline.

## Files Reviewed

- `packages/diffusion/src/families/flux/config.ts`
- `packages/diffusion/src/families/flux/latents.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/models/transformers/transformer_flux.py`
  defines the FLUX.1 `FluxTransformer2DModel` constructor shape, including 19
  dual-stream layers, 38 single-stream layers, 24 heads, 128 head dimension,
  4096 joint attention dimension, 768 pooled projection dimension, and
  `[16, 56, 56]` RoPE axes.
- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py`
  defines latent packing as 2x2 patches, inverse unpacking before VAE decode,
  and latent image ids shaped `[height * width, 3]`.
- `.reference/mlx-examples/flux/flux/flux.py` confirms the MLX Flux sampling
  path prepares packed latent images and image-position ids before transformer
  evaluation.
- `.reference/mlx-examples/flux/flux/model.py` confirms the Flux parameter
  shape used by the MLX example: 64 packed input channels, hidden size 3072,
  24 heads, 19 dual blocks, 38 single blocks, and 3-axis RoPE dimensions.
- The local partial FLUX.2-klein cache uses `Flux2Transformer2DModel`, 128
  input channels, four RoPE axes, and Flux2-only fields. The new parser rejects
  that shape instead of silently treating it as FLUX.1.

## Tensor Lifetime Audit

`packFluxLatents()` names the intermediate reshape and transpose tensors with
`using` bindings before returning the final packed reshape. `unpackFluxLatents()`
does the same for the inverse reshape and transpose path.

`createFluxLatentImageIds()` builds the id grid on the host as an `Int32Array`
and creates one returned `MxArray`. It does not create temporary tensor
intermediates.

`config.ts` parses JSON metadata only and does not allocate tensors.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/families/flux`: 9 pass, 0 fail.
- `bun run lint`: passed.
- `bun run --filter @mlxts/diffusion typecheck`: passed.

The tranche adds tensor shape helpers but no model hot path, no denoising loop,
and no benchmarkable generation path. It makes no image quality or throughput
claim.

## Independent Review

Faraday completed a read-only second pass before this artifact. The review
recommended shipping only FLUX.1 config parsing plus latent packing/id helpers
in this tranche, rejecting Flux2 shapes explicitly, keeping image ids unbatched,
and avoiding transformer or weight-loading work until the shape contract is
covered by tests.

## Remaining Risks / Follow-ups

- FLUX transformer blocks, RoPE application, guidance embedding, CLIP plus T5
  conditioning, VAE decode integration, checkpoint weight mapping, and real
  image proof remain follow-on Phase 10 tranches.
- FLUX.2, Qwen-Image, and Z-Image model families remain unsupported until their
  own reference audits and package-native shape contracts land.

## Out-of-scope drift noticed

The local Hugging Face cache contains partial FLUX.1-schnell lock files and a
partial FLUX.2-klein snapshot, but no complete local FLUX.1 checkpoint suitable
for end-to-end image validation in this tranche.
