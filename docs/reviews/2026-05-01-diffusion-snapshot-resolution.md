# Diffusion Snapshot Resolution Review

## Summary

`@mlxts/diffusion` now owns local-path and Hugging Face Hub snapshot resolution
for Diffusers checkpoints. The Stable Diffusion and FLUX proof CLIs accept a
local directory or model id, keep progress on stderr, and continue to load
generation components from a concrete local snapshot directory.

## Files Reviewed

- `packages/diffusion/src/pretrained/snapshot-source.ts`
- `packages/diffusion/src/index.ts`
- `examples/stable-diffusion/index.ts`
- `examples/flux/index.ts`

## Scope

This tranche adds snapshot-source ergonomics only. It does not change UNet,
VAE, transformer, scheduler, prompt-conditioning, tensor execution, sampling,
or image output semantics.

## Tensor Lifetime Audit

The resolver performs filesystem and Hugging Face Hub metadata/download work
only. It does not create `MxArray` values, hide disposable tensors inside nested
expressions, add native handles, or add MLX eval points. The example commands
still pass resolved local directories into the existing pipeline loaders.

## Memory / Performance Evidence

No generation hot path changed in this tranche. Focused validation before this
artifact:

- `bun test packages/diffusion/src/pretrained/snapshot-source.test.ts examples/stable-diffusion/index.test.ts examples/flux/index.test.ts`
- `bun run --filter '@mlxts/diffusion' typecheck`
- `bun run typecheck`
- `bun run check:file-lines`
- `bun run check:cross-package-imports`
- `bun run check:runtime-review`
- `bun run check:phase10-proofs`
- `bun test packages/diffusion`

All passed locally before this artifact was written.

## Independent Review

Descartes (`019de115-cc59-7e80-90b2-f315145f1446`) recommended proving Stable
Diffusion / SDXL before Qwen-Image, Z-Image, or FLUX.2 runtime work because SD
exercises the existing diffusion spine with the smallest architecture blast
radius. Popper (`019de11f-c2cf-7602-8fc2-28dd9631516f`) reviewed the final diff
and found two resolver edge cases: `@huggingface/hub` download debug output
could leak to stdout, and slash-containing revisions needed nested ref
directories. Both were fixed and covered by resolver tests before commit.

## Remaining Risks / Follow-ups

- Real SD/SDXL checkpoint image proof remains the next Phase 10 rung.
- The proof commands currently resolve or download snapshots inside the runtime
  command body. Moving network resolution before the heavy runtime lock is a
  later ergonomics tightening if downloads become long-running in practice.
- Remote selection intentionally downloads Diffusers metadata, tokenizer files,
  and safetensors only. Non-Diffusers conversion layouts such as mflux remain
  separate import work.

## Out-of-scope drift noticed

The local cache still lacks a complete runnable SD/SDXL or FLUX.1 snapshot, so
this tranche enables model-id/cache ergonomics but does not itself record a real
image proof.
