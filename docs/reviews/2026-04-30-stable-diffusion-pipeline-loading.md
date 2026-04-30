# Stable Diffusion Pipeline Loading

## Summary

Added `loadStableDiffusionPipelineFromSnapshot()` as the package-owned loading
boundary for Stable Diffusion runtime bundles. A local Diffusers snapshot now
loads into one disposable object containing the parsed manifest, parsed
component configs, VAE, UNet, and scheduler. Conditioning tensors remain supplied
by the caller; CLIP/tokenizer composition stays outside `@mlxts/diffusion`.
The bundle exposes thin runtime methods over the existing sampling helpers and
rejects enabled or required safety-checker semantics until that component is
owned explicitly.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/pipeline-loading.ts`
- `packages/diffusion/src/families/stable-diffusion/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  keeps pipeline construction distinct from prompt encoding and sampling
  inputs. The TypeScript boundary mirrors that separation by loading
  Diffusers-owned VAE, UNet, scheduler, and config truth while accepting
  conditioning tensors separately.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/__init__.py`
  constructs a model bundle from local weights before sampling. This tranche
  represents the same package-local component boundary without adding tokenizer
  or host image output concerns.

## Tensor Lifetime Audit

The loader itself does not create tensor intermediates beyond model parameter
loading. VAE and UNet weight loading remain delegated to the existing
shard-iterator loaders, which transfer assigned tensors into module parameters
and dispose skipped/error-path tensors. If any component fails after VAE load,
the partial bundle path disposes the loaded VAE before rethrowing. The returned
bundle disposes UNet and VAE exactly once. The adjacent CFG text-time path now
validates matching text-time tensor shapes before batching and frees the staged
`textEmbeds` concatenation if `timeIds` concatenation fails.

## Memory / Performance Evidence

This tranche adds a construction surface and makes no throughput claim. The
loader returns parsed metadata for the audit surface and reuses the existing
component loaders for tensor hydration. It does not retain conditioning tensors,
decoded images, prompt embeddings, or host image buffers. Caller-owned
conditioning tensors and returned image/latent tensors retain the existing
sampling ownership contract.

## Independent Review

Darwin reviewed the intended loader boundary before final edits. The review
recommended a small disposable runtime bundle, supplied tensor conditioning, no
`@mlxts/transformers` imports, idempotent disposal, post-dispose method guards,
partial-load cleanup, explicit safety-checker rejection, and cleanup for the
SDXL text-time concatenation edge. The implemented tranche incorporates those
points while keeping text conditioning as a follow-up.

## Validation

- `bun test packages/diffusion/src/families/stable-diffusion/pipeline-loading.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`

## Remaining Risks / Follow-ups

- The bundle deliberately stops before text conditioning. CLIP tokenization,
  prompt embedding composition, SDXL dual-encoder conditioning, and image
  encoding remain separate Phase 10 tranches.
- Real checkpoint image proof still needs a conditioning source and finite
  AXI-shaped command before diffusion is product-complete.
