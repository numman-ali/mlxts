# Runtime Review: LTX Latent Upsampler Runtime

## Summary

This tranche adds the classic LTX sidecar latent upsampler runtime. The package
now recognizes `LTXLatentUpsamplePipeline` manifests, parses and loads
`LTXLatentUpsamplerModel`, runs Diffusers-compatible spatial and temporal
PixelShuffleND paths, and exposes normalized BCFHW plus packed-latent
unpack/repack helpers.

This is a classic LTX tranche. LTX-2 rational resampling, 0.9.7
timestep-conditioned VAE decode, AdaIN/tone mapping, video artifact writing,
and finite checkpoint proof remain separate work.

## Files Reviewed

- `packages/diffusion/src/families/ltx/latent-upsampler.ts`
- `packages/diffusion/src/families/ltx/latent-upsample.ts`
- `packages/diffusion/src/families/ltx/latent-upsampler-weights.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`
- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/ltx-pipeline-specs.ts`
- `packages/diffusion/src/pretrained/snapshot-manifest.ts`
- `packages/diffusion/src/families/stable-diffusion/pipeline-loading.ts`

## Runtime Notes

The model boundary is normalized BCFHW latents. The low-level module converts to
BFHWC for Conv3d and flattens frames into BHWC batches for the Diffusers
spatial-only Conv2d upsampler path. Temporal upsampling drops the first frame
after PixelShuffleND, matching the classic Diffusers model.

Checkpoint loading maps Diffusers snake-case parameter names into the module
tree and translates Conv2d/Conv3d kernels from PyTorch layout to MLX
channel-last layout. The standalone directory loader supports sidecar repos
whose `config.json` and weights live directly in the model directory; the
snapshot loader supports `LTXLatentUpsamplePipeline` manifests with a
`latent_upsampler` component.

The root diffusion barrel delegates LTX exports through `src/ltx.ts` so the
package keeps a complete public LTX surface while preserving the file-line
gate.

## Tensor Lifetime Audit

The forward loops retain only the active hidden tensor and free the previous
hidden tensor after each block transition. Pixel shuffle helpers return retained
reshapes from scoped intermediates. Normalization helpers keep channel-stat
tensors scoped to the upsample call.

The safetensor loader frees skipped tensors, transformed source tensors, and
partially assigned tensors on failure. Snapshot and directory construction
dispose partially loaded models when any config, weight, or parameter
assignment step throws.

## Memory / Performance Evidence

No heavy checkpoint benchmark ran in this tranche. The runtime-sensitive shape
paths are covered synthetically: 1D/2D/3D PixelShuffleND order, spatial-only 3D
forward shape, temporal-spatial 3D frame trimming, 2D forward shape,
normalization round-trip, packed-latent repacking, Conv2d/Conv3d weight
translation, standalone directory loading, sidecar manifest loading, and strict
unexpected-weight failure.

## Validation

Focused LTX and manifest tests:

```bash
bun test packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/families/ltx
```

Result: 89 tests passed.

Focused diffusion typecheck:

```bash
bunx tsc -p packages/diffusion/tsconfig.json --pretty false
```

Result: passed.

File-line gate:

```bash
bun run check:file-lines
```

Result: passed.

## Independent Review

Peirce (`019de4c1-c4a5-79c1-8180-dd0ac1f1af64`) performed a read-only
second-opinion review against local Diffusers references and the current
Hugging Face sidecar layout. The review confirmed that classic LTX has a
separate `LTXLatentUpsamplerModel` / `LTXLatentUpsamplePipeline`, that LTX-2's
rational resampler is a separate class, and that this tranche should focus on
classic-only config/loading, PixelShuffleND parity, normalized latent
round-trip, packed helper coverage, and sidecar snapshot recognition.

## Out-of-scope Drift Noticed

Classic LTX 0.9.7 VAE configs still require timestep-conditioned decoder paths
and residual/factorized upsampling before a complete 0.9.7 proof can run.

## Remaining Risks / Follow-ups

The new upsampler has synthetic runtime and loader coverage but no bounded real
sidecar checkpoint proof yet. The next LTX tranche should add
`examples/ltx-video` proof orchestration and artifact evidence, then revisit
the 0.9.7 decoder deltas needed for newer official checkpoints.
