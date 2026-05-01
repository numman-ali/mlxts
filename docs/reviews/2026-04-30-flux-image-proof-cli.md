# FLUX Image Proof CLI Runtime Review

## Files Reviewed

- `packages/diffusion/src/families/flux/autoencoder.ts`
- `packages/diffusion/src/families/flux/config.ts`
- `packages/diffusion/src/index.ts`
- `examples/flux/index.ts`
- `examples/flux/image-output.ts`

## Summary

This tranche adds a FLUX AutoencoderKL wrapper, VAE config coverage, FLUX VAE
weight loading, and a local image proof CLI. The command loads a Diffusers FLUX
snapshot, validates the pipeline kind, uses the FlowMatch Euler scheduler,
encodes CLIP/T5 prompt conditioning, denoises through the FLUX transformer, and
writes a BMP artifact.

## Tensor Lifetime Audit

The FLUX VAE wrapper reuses the Stable Diffusion AutoencoderKL module topology,
so its tensor ownership follows the existing module and weight-loader rules.
Loaded model parameters are evaluated once after assignment. The CLI scopes
transformer, VAE, prompt conditioner, conditioning tensors, RNG keys, and final
image tensors with explicit disposal.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/families/flux`: 32 pass.
- `bun test examples/flux`: 15 pass.
- `bunx tsc -p tsconfig.phase10-examples.json`: pass.
- `bun run typecheck`: pass.
- `bun run check:tensor-lifetimes`: pass.
- `bun run check:runtime-review`: pass.
- `bun run validate`: pass.

## Independent Review

Read-only second review agreed that FLUX.1 VAE should remain a thin
AutoencoderKL semantic wrapper over the existing VAE topology, with FLUX-owned
latent shift metadata and Diffusers config validation.

## Remaining Risks / Follow-ups

Full visual proof still depends on a local FLUX checkpoint and remains a heavy
manual/runtime QA step outside the default test gate.
