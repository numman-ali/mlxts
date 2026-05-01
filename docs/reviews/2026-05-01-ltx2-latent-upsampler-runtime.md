# Runtime Review: LTX-2 Latent Upsampler Runtime

## Summary

This tranche adds the LTX-2 sidecar latent upsampler foundation. The package now
recognizes `LTX2LatentUpsamplePipeline` snapshots, selects remote
`latent_upsampler` artifacts, parses and loads `LTX2LatentUpsamplerModel`, and
runs BCFHW latent upsampling with Diffusers-compatible rational spatial
resampling.

This is not the full LTX-2 audio-video denoising path. Gemma3 hidden-state
conditioning, `LTX2TextConnectors`, dual-stream audio/video transformer
execution, LTX-2 video/audio VAE decode, vocoder output, and MP4/WAV artifact
writing remain separate tranches.

## Files Reviewed

- `packages/diffusion/src/families/ltx/latent-upsampler-ltx2.ts`
- `packages/diffusion/src/families/ltx/latent-upsample-ltx2.ts`
- `packages/diffusion/src/families/ltx/latent-upsampler-ltx2-weights.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`
- `packages/diffusion/src/pretrained/ltx-pipeline-specs.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/snapshot-file-selection.ts`

## Reference Audit

Diffusers current LTX-2 sidecar uses `LTX2LatentUpsamplePipeline` with
`LTX2LatentUpsamplerModel`, a classic-style residual upsampler, optional
temporal/spatial pixel shuffle paths, and the newer rational spatial resampler.
The TypeScript runtime keeps the same model-level boundary: BCFHW latents enter
the module, spatial-only 3D execution flattens per-frame work through a 2D
resampler, temporal upsampling drops the first generated frame, and packed-token
helpers unpack/repack around the sidecar.

The fixed blur-downsample kernel is generated as a non-parameter tensor at
runtime, matching Diffusers' registered buffer semantics instead of making it a
trainable/load-required parameter.

## Tensor Lifetime Audit

Forward paths use explicit `using` scopes for intermediate transposes,
reshapes, convolutions, pixel shuffle tensors, blur kernels, and latent
normalization tensors. Mutable loop temporaries free the previous hidden tensor
before retaining the next value. Weight loading frees skipped tensors,
transformed source tensors, and failed partial assignments on all error paths.

The package still keeps local tensor lifetimes visible in the runtime code; no
new nested tensor-producing expression hides disposable `MxArray` ownership.

## Memory / Performance Evidence

No performance claim is made. The new runtime adds tensor execution coverage for
tiny synthetic inputs only. The hot-path choice is conservative: rational
spatial resampling uses MLX `conv2d` with depthwise groups and fixed generated
kernel weights rather than host-side resizing.

Focused validation:

```bash
bunx tsc -p packages/diffusion/tsconfig.json --pretty false
bun test packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/pretrained/snapshot-file-selection.test.ts packages/diffusion/src/families/ltx
```

Result: `111 pass`, `0 fail`.

## Independent Review

Noether reviewed the next Phase 10 LTX direction independently before the
implementation was finalized. The review confirmed that LTX-2 is a distinct
audio-video family, that full denoising requires Gemma3 hidden-state
conditioning plus connectors and dual-stream transformer work, and that
`LTX2LatentUpsamplePipeline` recognition was the immediate local inconsistency
to close before broader LTX-2 runtime work.

## Out-of-scope Drift Noticed

Some older Phase 10 review prose still describes Stable Diffusion 3 as
runtime-incomplete. `PLAN.md`, `packages/diffusion/README.md`, and the SD3 proof
artifacts now supersede that older review wording.

## Remaining Risks / Follow-ups

- Full LTX-2 checkpoint generation is still blocked on connector, transformer,
  video/audio VAE, and vocoder runtime work.
- LTX-2 audio output will likely require package-owned ConvTranspose1d-style
  support before the vocoder can be faithful.
- The current LTX-2 sidecar path has synthetic shape and loading coverage, not
  an authenticated real checkpoint proof.
