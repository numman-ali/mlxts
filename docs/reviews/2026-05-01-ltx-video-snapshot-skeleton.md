# Runtime Review: LTX Video Snapshot Skeleton

## Summary

This tranche adds manifest-only recognition for current Diffusers LTX-Video and
LTX-2 snapshots. `@mlxts/diffusion` now parses `LTXPipeline`,
`LTXConditionPipeline`, and `LTX2Pipeline` model indexes into package-owned
pipeline kinds and component roles, without constructing runtime video/audio
models or importing transformer encoders.

The upstream manifests reviewed were the live Hugging Face
`Lightricks/LTX-Video`, `Lightricks/LTX-Video-0.9.7-dev`, and
`Lightricks/LTX-2` Diffusers `model_index.json` layouts, plus the current
Diffusers LTX-2 pipeline docs.

## Files Reviewed

- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/ltx-pipeline-specs.ts`
- `packages/diffusion/src/pretrained/snapshot-file-selection.ts`

## Tensor Lifetime Audit

No tensor-producing code changed. The touched package files parse JSON metadata
and component file requirements only. No `MxArray`, scheduler step, model
forward, safetensor tensor load, or disposal path was introduced.

## Memory / Performance Evidence

No generation hot path changed, and no benchmark claim is made. The focused
metadata gate passed:

```bash
bun test packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/pretrained/snapshot-file-selection.test.ts
```

Result: 25 tests passed, including LTX-Video and LTX-2 parse, local snapshot
manifest coverage, and remote LTX-2 component folder selection.

## Independent Review

Boole performed a read-only second-opinion review against the repo pretraining
surface and upstream LTX manifests. The review caught that Hub snapshot
selection would have skipped required LTX-2 `audio_vae`, `connectors`, and
`vocoder` component folders, and that the base `LTXPipeline` path needed direct
coverage. This artifact reflects the follow-up fixes.

## Remaining Risks / Follow-ups

This is not runtime video/audio generation. LTX transformer execution, video
VAE execution, LTX-2 audio VAE/vocoder execution, latent upsampling, artifact
encoding, and AXI proof commands remain future Phase 10 tranches.
