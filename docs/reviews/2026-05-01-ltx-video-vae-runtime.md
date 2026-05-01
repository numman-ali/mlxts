# Runtime Review: LTX Video VAE Runtime

## Summary

This tranche adds the classic LTX-Video decoder-side VAE runtime needed after
packed transformer denoising. The package now unpacks packed LTX video latents,
applies Diffusers channelwise latent denormalization, runs a decoder-only
`AutoencoderKLLTXVideo` surface, and returns BFHWC video tensors in the `0..1`
range.

This is a decode-first tranche. It deliberately does not add VAE encode,
posterior sampling, tiling/slicing, the 0.9.7 timestep-conditioned decoder,
latent upsampler execution, video file encoding, LTX-2 audio/video runtime, or
text encoder orchestration.

## Files Reviewed

- `packages/diffusion/src/families/ltx/autoencoder.ts`
- `packages/diffusion/src/families/ltx/autoencoder-blocks.ts`
- `packages/diffusion/src/families/ltx/autoencoder-volume.ts`
- `packages/diffusion/src/families/ltx/autoencoder-weights.ts`
- `packages/diffusion/src/families/ltx/decoding.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/index.ts`

## Runtime Notes

The decoder keeps the public VAE boundary as BCFHW, matching Diffusers LTX
latents. Internally, Conv3d runs over BFHWC because MLX convolution kernels are
channel-last. Checkpoint Conv3d kernels are translated from Diffusers
`[out, in, kT, kH, kW]` to MLX `[out, kT, kH, kW, in]`.

Classic LTX VAE RMSNorm is parameterless. The implementation uses fused RMSNorm
without learnable weights and keeps the only affine normalization path at the
Diffusers `norm3` shortcut LayerNorm.

Latent stats are checkpoint buffers, not normal module parameters. The loader
consumes `latents_mean` and `latents_std` separately and intentionally skips
encoder weights because this tranche owns generation decode only.

Unsupported variants fail at construction: timestep-conditioned decode,
temporal VAE patching beyond `patch_size_t = 1`, decoder noise injection,
residual upsampling, and non-1 decoder upsample factors.

## Tensor Lifetime Audit

New tensor-producing paths use `using` scopes for temporaries and explicit
retain only when returning values from scoped intermediates. Long-lived model
state is limited to module parameters and loaded latent-stat arrays.

The decoder loop frees the previous hidden tensor after each block transition.
The checkpoint loader frees skipped tensors, transformed source tensors, and
partially assigned tensors on failure.

## Memory / Performance Evidence

This tranche did not run a heavy checkpoint benchmark. It adds synthetic runtime
coverage for the memory-sensitive shape and ownership paths: temporal edge
padding, Conv3d kernel translation, decoder unpatch order, latent-stat
denormalization, and loader tensor disposal.

## Validation

Focused LTX VAE tests:

```bash
bun test packages/diffusion/src/families/ltx/autoencoder.test.ts packages/diffusion/src/families/ltx/autoencoder-volume.test.ts packages/diffusion/src/families/ltx/autoencoder-weights.test.ts packages/diffusion/src/families/ltx/decoding.test.ts
```

Result: 18 tests passed.

Broader LTX family gate:

```bash
bun test packages/diffusion/src/families/ltx
```

Result: 63 tests passed.

Focused diffusion typecheck:

```bash
bunx tsc -p packages/diffusion/tsconfig.json --pretty false
```

Result: passed.

## Independent Review

Popper (`019de49f-4c63-7392-99cc-90a43d960b7a`) performed a read-only
second-opinion review against current Diffusers LTX references and local VAE
patterns. The review recommended a decoder-first `AutoencoderKLLTXVideo`
surface, explicit encoder-weight skipping, Diffusers latent denormalization
order, Conv3d kernel transposition, parameterless RMSNorm, and tests for
temporal padding, unpatch order, weight mapping, and latent stats.

## Out-of-scope Drift Noticed

`Lightricks/LTX-Video-0.9.7-dev` uses `LTXVideo095DownBlock3D`,
timestep-conditioned decoder paths, residual/factorized upsampling, and latent
upsampling. Those remain separate runtime tranches.

## Remaining Risks / Follow-ups

The new decoder has synthetic layout and loader coverage, but no bounded real
LTX checkpoint proof has run through transformer plus VAE decode yet. The next
tranche needs the finite `examples/ltx-video` proof command and saved artifact
evidence.
