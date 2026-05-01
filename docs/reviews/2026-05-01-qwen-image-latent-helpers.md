# Runtime Review: Qwen-Image Latent Helpers

## Summary

Added the package-owned Qwen-Image latent layout foundation: NCFHW initial latent shape, Diffusers-compatible 2x2 patch packing, packed-sequence unpacking, RoPE image-shape derivation, and scheduler initial latent creation. The tranche only establishes the tensor layout contract needed by the later Qwen-Image transformer and pipeline work.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/latents.ts`
- `packages/diffusion/src/families/qwen-image/latents.test.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

`packQwenImageLatents`, `unpackQwenImageLatents`, and `createQwenImageInitialLatents` keep reshape/transpose/sample intermediates visible with `using` declarations and return only the retained output tensor. Shape validators are host-only and do not allocate MLX tensors. `bun run check:tensor-lifetimes` passed.

## Memory / Performance Evidence

Focused tests cover latent shape divisibility, packed sequence shape, single-channel and multi-channel Diffusers patch order, rank-4 pachifier parity, unpack round-trip shape, scheduler-created initial latents, RoPE image-shape derivation, and malformed-shape rejection. `bun run validate` passed. No real checkpoint image generation or performance claim is made in this tranche.

## Independent Review

Meitner (`019de1be-ea57-7651-9dd0-02f06fd0be85`) performed a read-only Qwen-Image reference audit against local Diffusers. The review corrected the tranche to Qwen-Image's channel-first latent layout, identified `QwenImagePachifier.pack_latents` / `unpack_latents` as the source of truth, and called out Conv3d as a prerequisite for full VAE decode.

## Remaining Risks / Follow-ups

Qwen-Image transformer blocks, RoPE frequencies, prepared-prompt denoising, VAE decode, safetensor weight mapping, and real checkpoint image proof remain future tranches. The Qwen VAE requires a proper `mlx_conv3d` binding and `@mlxts/nn` Conv3d layer before full decode support can be claimed; no JS fallback belongs on that path.

## Out-of-scope Drift Noticed

None.
