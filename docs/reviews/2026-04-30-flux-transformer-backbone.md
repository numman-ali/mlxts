# FLUX Transformer Backbone Runtime Review

## Files Reviewed

- `packages/diffusion/src/families/flux/attention.ts`
- `packages/diffusion/src/families/flux/blocks.ts`
- `packages/diffusion/src/families/flux/embeddings.ts`
- `packages/diffusion/src/families/flux/pipeline.ts`
- `packages/diffusion/src/families/flux/tensor-utils.ts`
- `packages/diffusion/src/families/flux/transformer.ts`
- `packages/diffusion/src/index.ts`

## Summary

This tranche adds the FLUX.1 transformer denoiser tensor path: normalized
flow-time input, timestep/vector conditioning, ND RoPE, double-stream joint
attention, single-stream transformer blocks, and final AdaLN projection.

## Tensor Lifetime Audit

Intermediate tensor-producing calls are bound to visible locals with explicit
resource scopes. Split outputs and returned projection/modulation objects have
dedicated disposal paths.

## Memory / Performance Evidence

Local validation:

- `bun test packages/diffusion/src/families/flux/transformer.test.ts`
- `bun test packages/diffusion/src/families/flux/pipeline.test.ts`
- `bun test packages/diffusion/src/families/flux`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`

## Independent Review

The FLUX backbone audit sub-agent reviewed Diffusers and MLX example semantics.
The review flagged raw scheduler timestep input as the main mismatch; this
tranche now passes normalized scheduler sigma into the denoiser and keeps the
1000x multiplier inside the FLUX timestep embedding.

## Remaining Risks / Follow-ups

Weight-name mapping is not part of this tranche. Numeric parity against
Diffusers/MLX examples begins after real checkpoint loading lands.
