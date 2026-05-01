# Qwen-Image VAE Loading Runtime Review

## Summary

Qwen-Image now has VAE safetensor loading from local Diffusers snapshot
manifests and a bounded decode helper for packed latent tensors. This tranche
proves VAE component loading and latent decode plumbing only. It does not claim
Qwen-Image transformer execution, text conditioning, denoising, or generated
image quality.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/weights.ts`
- `packages/diffusion/src/families/qwen-image/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Runtime Sensitivity

The changed files assign real MLX array parameters from safetensor shards,
transpose Conv3d and Conv2d checkpoint tensors, evaluate loaded VAE parameters,
and decode packed latent tensors through Qwen-specific per-channel mean/std
normalization. Incorrect ownership, layout, or decode scaling would corrupt VAE
execution or leak loaded arrays.

## Tensor Lifetime Audit

Skipped and unexpected checkpoint tensors are freed immediately. Transformed
checkpoint tensors free their original source when the transform returns a new
array. Assignment frees the previous model parameter only after shape validation,
then transfers ownership to the module field. Error paths free the pending
assigned tensor.

Latent decode keeps unpack, mean/std reshape, scale, shift, raw decode, frame
slice, channel transpose, normalization, and clamp intermediates in visible
`using` declarations. The returned image tensor is the only live output.

## Reference Parity

Diffusers Qwen-Image decodes packed latents by unpacking 2x2 patches into a
single-frame NCFHW latent volume, applying `latents * latents_std + latents_mean`,
running `AutoencoderKLQwenImage.decode`, selecting frame `0`, and postprocessing
from `[-1, 1]` to image range. The helper mirrors that base decode tail for
prepared packed latents.

Checkpoint transforms preserve Diffusers-to-MLX layout parity: Conv3d weights
map `[out, in, kT, kH, kW]` to `[out, kT, kH, kW, in]`, Conv2d weights map
`[out, in, kH, kW]` to `[out, kH, kW, in]`, and RMS `gamma` singleton tails
squeeze to `[channels]`.

## Memory / Performance Evidence

Focused tests cover complete tiny VAE safetensor loading, indexed shard loading,
snapshot-manifest construction, missing weights, shape mismatches, strict
unexpected weights, decode shape/postprocess, and mean/std application.

No performance optimization claim is made. Real checkpoint image decode remains
pending until transformer, conditioning, denoising, and full VAE parity are
landed and tested together.

## Independent Review

Hubble the 2nd reviewed the Qwen-Image checkpoint-support boundary and
recommended VAE loading plus decode-from-packed-latents as the smallest safe
tranche. The transformer runtime, text encoder conditioning, CFG, and CLI proof
remain out of scope.

## Remaining Risks / Follow-ups

The local Qwen VAE raw decoder still omits Diffusers temporal chunk cache,
tiling, and slicing behavior. This is acceptable for the single-frame base
image decode helper, but multi-frame/video parity is not claimed.

Qwen-Image transformer support and Qwen2.5-VL prompt conditioning are still
required before a real text-to-image proof can run.

## Out-of-scope Drift Noticed

None.
