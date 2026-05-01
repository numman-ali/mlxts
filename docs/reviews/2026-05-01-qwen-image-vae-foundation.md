# Qwen-Image VAE Foundation Runtime Review

## Summary

Qwen-Image now has VAE building blocks, a channel-first public tensor boundary,
an internal channel-last MLX Conv3d path, and Diffusers-to-MLX VAE weight layout
transforms. This is a foundation tranche, not a full real-checkpoint decode
claim.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/autoencoder-blocks.ts`
- `packages/diffusion/src/families/qwen-image/autoencoder-volume.ts`
- `packages/diffusion/src/families/qwen-image/autoencoder.ts`
- `packages/diffusion/src/families/qwen-image/weights.ts`
- `packages/diffusion/src/index.ts`

## Runtime Sensitivity

This tranche adds Qwen-Image VAE tensor modules on top of MLX Conv3d, spatial
attention, tensor layout transposes, causal temporal padding, and checkpoint
weight layout transforms. It is runtime-sensitive because incorrect layout or
ownership would corrupt image latents or leak MLX arrays during VAE execution.

## Tensor Lifetime Audit

Intermediate tensor-producing calls are held in visible `using` declarations or
explicit `try`/`finally` ownership blocks. The causal Conv3d wrapper frees padded
and prefixed volumes after the convolution graph is built. Encoder, decoder,
mid-block, and up-block loops free the previous hidden state immediately after a
replacement tensor is produced, and free the current hidden state on error.

The QKV `split` outputs in the attention block are explicitly freed in a
`finally` block because they are returned as an array rather than bound by
individual `using` declarations at creation.

## Reference Parity

Diffusers `AutoencoderKLQwenImage` uses public channel-first `[B, C, T, H, W]`
volumes, causal temporal left padding, and PyTorch Conv3d weights shaped
`[out, in, kT, kH, kW]`. The package boundary keeps public VAE tensors in NCFHW
and converts the internal MLX path to NDHWC. Conv3d weights transpose to
`[out, kT, kH, kW, in]`, and Conv2d weights transpose to `[out, kH, kW, in]`.

## Memory / Performance Evidence

Focused Qwen-Image tests cover layout conversion, causal padding, prefix padding
reduction, resample shape behavior, tiny VAE construction, NCFHW encode/decode
boundaries, and weight path/layout transforms.

No performance optimization claim is made. The tranche preserves visible tensor
ownership around Conv3d, pad, transpose, reshape, concatenate, split, and spatial
attention intermediates.

## Independent Review

Kierkegaard the 2nd reviewed the Qwen-Image VAE reference shape and recommended
landing VAE architecture plus weight-layout foundation without claiming real
checkpoint image decode. The implementation follows that boundary.

## Remaining Risks / Follow-ups

Real checkpoint decode is not claimed in this tranche. The next tranche needs
full Qwen-Image VAE checkpoint loading, decode parity against a known image
latent fixture, and a memory review of attention and temporal cache behavior.

## Out-of-scope Drift Noticed

None.
