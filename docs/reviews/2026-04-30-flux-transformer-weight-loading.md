# FLUX Transformer Weight Loading Runtime Review

## Files Reviewed

- `packages/diffusion/src/families/flux/blocks.ts`
- `packages/diffusion/src/families/flux/weight-mapping.ts`
- `packages/diffusion/src/families/flux/weights.ts`
- `packages/diffusion/src/index.ts`

## Summary

This tranche adds Diffusers FLUX.1 transformer weight mapping and safetensors
loading. Diffusers split q/k/v tensors are fused into package-owned packed
projection parameters at load time, preserving the runtime projection layout.
The single-stream block now owns the Diffusers Q/K RMSNorm parameters required
for strict FLUX.1 transformer loads, and the loader translates Diffusers final
AdaLN scale/shift ordering into the package-owned shift/scale runtime order.

## Tensor Lifetime Audit

Safetensor tensors are either assigned into the module, retained temporarily as
fused projection parts, or freed immediately as unexpected weights. Pending
fused parts are released in a `finally` block after assignment or failure.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/families/flux/weights.test.ts`: 5 pass.
- `bun test packages/diffusion/src/families/flux`: 28 pass.
- `bun run typecheck`: pass.
- `bun run check:tensor-lifetimes`: pass.
- `bun run check:runtime-review`: pass.
- `bun run validate`: pass.

## Independent Review

Read-only FLUX weight-mapping review found two parity requirements that are now
covered here: single-stream Q/K RMSNorm parameters and Diffusers
`norm_out.linear` scale/shift half ordering.

## Remaining Risks / Follow-ups

Full numeric parity begins after the FLUX proof CLI can load a real checkpoint
and run denoising evidence against a known prompt.
