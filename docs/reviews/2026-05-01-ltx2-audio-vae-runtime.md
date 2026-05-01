# Runtime Review: LTX-2 Audio VAE Runtime

## Summary

This tranche adds the LTX-2 `AutoencoderKLLTX2Audio` decode side in
`@mlxts/diffusion`. The new surface covers decoder-only execution, BLMC
package-internal Conv2d flow, BCLM public tensor boundaries, packed-token
latent denormalization, Diffusers safetensor loading, and cross-component
dimension checks for the audio VAE and vocoder boundary.

The scope is intentionally audio-VAE-only. LTX-2 vocoder execution, prompt
encoding, full pipeline assembly, waveform output, and LTX-2.3 branches remain
separate Phase 10 tranches.

## Files Reviewed

- `packages/diffusion/src/families/ltx/autoencoder-ltx2-audio.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-audio-blocks.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-audio-weights.ts`
- `packages/diffusion/src/families/ltx/config.ts`
- `packages/diffusion/src/families/ltx/decoding.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-audio.test.ts`
- `packages/diffusion/src/families/ltx/autoencoder-ltx2-audio-weights.test.ts`
- `packages/diffusion/src/families/ltx/config.test.ts`

## Tensor Lifetime Audit

Decoder execution keeps tensor-producing operations in local bindings. Causal
padding, Conv2d calls, residual branches, attention projections, upsampling,
normalization, activation, and crop/pad outputs use `using` bindings or
explicit `try/finally` ownership when a mutable `hidden` reference is replaced.
Returned tensors are retained or transferred deliberately before local
intermediates are freed.

Audio latent denormalization applies Diffusers' packed-token stat contract
before BCLM unpacking. Stat tensors are lexical `using` values. The weight
loader preserves the package shard-iterator ownership pattern: assigned tensors
move into the module tree, transformed source tensors are freed, skipped
tensors are freed immediately, latent-stat tensors are copied to host arrays
and freed, and partial error paths dispose candidate tensors.

## Memory / Performance Evidence

This is a new decode and loading capability, not a hot-path optimization claim.
Checkpoint loading uses `iterateSafetensors` and does not materialize whole
shards eagerly.

Focused validation:

```bash
bun test packages/diffusion/src/families/ltx/autoencoder-ltx2-audio.test.ts packages/diffusion/src/families/ltx/autoencoder-ltx2-audio-weights.test.ts packages/diffusion/src/families/ltx/config.test.ts
```

Result: 22 tests passed.

Adjacent LTX-2 video/decode regression:

```bash
bun test packages/diffusion/src/families/ltx/autoencoder-ltx2.test.ts packages/diffusion/src/families/ltx/autoencoder-ltx2-weights.test.ts packages/diffusion/src/families/ltx/decoding.test.ts
```

Result: 16 tests passed.

Type validation:

```bash
bun run typecheck
```

Result: passed.

Full LTX family regression after fixture tightening:

```bash
bun test packages/diffusion/src/families/ltx
```

Result: 143 tests passed.

Coverage gate:

```bash
bun run check:coverage
```

Result: passed. `@mlxts/diffusion` reported 95.06% line coverage and 93.86%
function coverage.

## Independent Review

Descartes the 2nd completed a read-only second pass against local Diffusers
references and the in-progress package code. The review confirmed the default
LTX-2 path and called out three hidden-contract risks: latent stat buffers are
`base_channels`, not an arbitrary packed-feature alias; vocoder input width
comes from decoded audio channels times mel bins; decoder causal cropping is
keyed by `causality_axis is not None`, not `is_causal`.

Those findings were folded into this tranche. `Ltx2AudioAutoencoderKL` now uses
`baseChannels` as the stat length, LTX-2 config loading checks
`baseChannels`, packed audio feature width, transformer audio channels, and
vocoder decoded-audio width explicitly, and decoder output cropping follows
`causalityAxis !== null`.

Reference files checked:

- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2_audio.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/vocoder.py`

## Remaining Risks / Follow-ups

- LTX-2 vocoder execution and loading remain required before audio latents can
  become waveform output.
- LTX-2 full pipeline assembly still needs prompt encoding, connector
  invocation, component bundle loading, video decode, audio decode, vocoder
  output, and artifact writing.
- LTX-2.3 prompt modulation, gated/perturbed attention, STG, and modality
  isolation remain explicitly unsupported until their full runtime paths land.

## Out-of-scope Drift Noticed

None.
