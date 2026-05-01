# Runtime Review: LTX Video Component Configs

## Summary

This tranche adds typed component config parsing for the LTX-Video and LTX-2
snapshot families already recognized by `@mlxts/diffusion`. The package now
loads current Diffusers LTX component configs into package-owned shapes for
LTX-Video transformer/VAE and LTX-2 transformer, video VAE, audio VAE, text
connectors, and vocoder. It does not add runtime video/audio model execution,
text encoder imports, artifact writing, or proof commands.

The upstream configs reviewed were the live Hugging Face `Lightricks/LTX-Video`
and `Lightricks/LTX-2` Diffusers component configs, plus local Diffusers
references for `LTXVideoTransformer3DModel`, `AutoencoderKLLTXVideo`,
`LTX2VideoTransformer3DModel`, `AutoencoderKLLTX2Video`,
`AutoencoderKLLTX2Audio`, `LTX2TextConnectors`, and `LTX2Vocoder`.

## Files Reviewed

- `packages/diffusion/src/families/flux2/config-parsing.ts`
- `packages/diffusion/src/families/ltx/config-common.ts`
- `packages/diffusion/src/families/ltx/config-ltx2-autoencoders.ts`
- `packages/diffusion/src/families/ltx/config-ltx2-media.ts`
- `packages/diffusion/src/families/ltx/config-ltx2-transformer.ts`
- `packages/diffusion/src/families/ltx/config-ltx2.ts`
- `packages/diffusion/src/families/ltx/config.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

No tensor-producing code changed. The new files parse JSON metadata, validate
component dimensions, and expose config records. No `MxArray`, scheduler step,
model forward, safetensor tensor load, or disposal path was introduced.

## Memory / Performance Evidence

No generation hot path changed, and no performance claim is made. The focused
metadata gate passed:

```bash
bun test packages/diffusion/src/families/ltx/config.test.ts
```

Result: 6 tests passed, covering LTX-Video parsing, LTX-2 parsing, manifest
loading, unsupported variants, and cross-component mismatch rejection.

`bun run typecheck` also passed after the parser and export surface changed.

## Independent Review

Boole performed a read-only second-opinion review against package doctrine,
existing diffusion config parsers, local Diffusers LTX references, and current
Hugging Face LTX configs. The review agreed that `families/ltx/` is the right
package-local home and caught the LTX-2 audio gotcha: the transformer audio
input check is against packed audio feature width, not raw audio VAE latent
channels. The landed parser includes that cross-check.

## Remaining Risks / Follow-ups

This is still not runtime video/audio generation. LTX transformer execution,
video VAE execution, LTX-2 audio VAE/vocoder execution, latent upsampling,
artifact encoding, and AXI proof commands remain future Phase 10 tranches.

LTX-Video 0.9.7+ latent upsampler and image/video conditioning pipeline variants
remain intentionally outside this tranche until their runtime semantics are
designed as separate package-owned surfaces.
