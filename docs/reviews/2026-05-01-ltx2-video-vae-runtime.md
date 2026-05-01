# Runtime Review: LTX-2 Video VAE Runtime

## Summary

This tranche adds the LTX-2 `AutoencoderKLLTX2Video` decode side in
`@mlxts/diffusion`. The new surface covers decoder-only execution, LTX-2
residual pixel-shuffle upsampling, reflect spatial padding, channelwise latent
statistics, and Diffusers safetensor loading for the video VAE component.

The scope is intentionally video-only. LTX-2 audio VAE execution, vocoder
execution, prompt encoding, full pipeline assembly, and LTX-2.3 branches remain
separate Phase 10 tranches.

## Files Reviewed

- `packages/diffusion/src/families/ltx/autoencoder-ltx2.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-blocks.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-weights.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2.test.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-weights.test.ts`

## Tensor Lifetime Audit

Decoder execution keeps every tensor-producing operation in a local binding.
The causal convolution path owns temporal padding, spatial padding, Conv3d, and
residual pixel-shuffle tensors with `using` or explicit `try/finally` lifetime
management. The weight loader preserves the existing shard-iterator ownership
pattern: assigned tensors move into the module tree, transformed source tensors
are freed, skipped tensors are freed immediately, and partially assigned error
paths dispose the candidate tensor.

The reflect-padding helper creates shape-local index tensors per padded axis.
Those tensors are lexical `using` values and do not escape the call.

## Memory / Performance Evidence

This is a new LTX-2 decode capability, not a hot-path optimization claim. The
implementation avoids whole-shard eager materialization by using
`iterateSafetensors`, matching the existing diffusion loading posture.

Focused validation:

```bash
bun test packages/diffusion/src/families/ltx/autoencoder-ltx2.test.ts packages/diffusion/src/families/ltx/autoencoder-ltx2-weights.test.ts
bun run typecheck
bun run lint
bun run check:file-lines
```

Final validation:

```bash
bun run validate
```

## Independent Review

Faraday the 2nd completed a read-only LTX-2 reference audit before this tranche.
The review recommended landing video VAE decode and loading before audio
VAE/vocoder work, called out LTX-2 residual upsampling and `upsample_factor: 2`,
plain Conv3d shortcuts, 5D Conv3d weight transposition, latent stat buffers, and
Diffusers reflect spatial padding as the main correctness traps.

Reference files checked:

- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py`
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2_audio.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/vocoder.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`

## Remaining Risks / Follow-ups

- LTX-2 audio VAE decode and `LTX2Vocoder` execution/loading remain required for
  audio-video product output.
- LTX-2 full pipeline assembly still needs prompt encoding, connector
  invocation, component bundle loading, video decode, audio decode, vocoder
  output, and artifact writing.
- LTX-2.3 prompt modulation, gated/perturbed attention, STG, and modality
  isolation remain explicitly unsupported until their full runtime paths land.
- Reflect spatial padding is implemented package-locally with `takeAxis`
  indices because MLX pad exposes `constant` and `edge`, not PyTorch-style
  `reflect`. A later backend helper can reduce per-call index construction if
  real decode profiling shows it matters.
