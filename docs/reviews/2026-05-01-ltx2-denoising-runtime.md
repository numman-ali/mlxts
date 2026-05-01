# Runtime Review: LTX-2 Denoising Runtime

## Summary

This tranche adds the prepared-conditioning LTX-2 audio-video denoising loop.
The runtime now accepts packed video latents, packed audio latents, connector
embeddings for both modalities, and a paired denoiser that returns video and
audio velocity predictions.

This is not the full LTX-2 product path. Gemma3 hidden-state extraction,
`LTX2TextConnectors`, the full dual-stream transformer module, video/audio VAE
decode, vocoder output, and final artifact writing remain separate tranches.

## Files Reviewed

- `packages/diffusion/src/families/ltx/pipeline-ltx2.ts`
- `packages/diffusion/src/families/ltx/pipeline-ltx2-types.ts`
- `packages/diffusion/src/families/ltx/pipeline-ltx2-validation.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`

## Reference Audit

Diffusers current `LTX2Pipeline` prepares video and audio latents, resolves a
shared schedule for both modalities, precomputes video and audio coordinate
tensors, duplicates those coordinates for classifier-free guidance, and calls
the transformer once for paired negative/positive predictions.

The LTX-2 guidance rule is modality-specific and happens in denoised `x0`
space. The TypeScript runtime mirrors that boundary: velocity predictions are
converted to denoised samples per modality, CFG deltas are applied to the video
and audio denoised predictions independently, and the guided denoised samples
are converted back to velocities before each scheduler step.

STG, modality isolation guidance, and guidance rescale each require additional
Diffusers forward paths. This tranche rejects non-default values for those
options instead of accepting partially modeled behavior.

## Tensor Lifetime Audit

The sampling loop retains caller-owned initial latents, frees the prior current
latents only after both next modality latents are produced, and frees paired
denoiser predictions after conversion to scheduler velocities. Guided
conditioning tensors and repeated coordinate tensors have explicit disposal on
success and error paths.

No nested tensor-producing expression hides disposable `MxArray` ownership.

## Memory / Performance Evidence

No performance claim is made. The new loop adds semantic runtime structure and
synthetic unit coverage only. The implementation keeps one denoiser forward per
step for CFG, matching the Diffusers combined negative/positive batch path for
the supported guidance mode.

Focused validation:

```bash
bun run --filter @mlxts/diffusion typecheck
bun run lint -- packages/diffusion/src/families/ltx/pipeline-ltx2.ts packages/diffusion/src/families/ltx/pipeline-ltx2.test.ts packages/diffusion/src/families/ltx/index.ts packages/diffusion/src/ltx.ts
bun test packages/diffusion/src/families/ltx/pipeline-ltx2.test.ts packages/diffusion/src/families/ltx/latents.test.ts packages/diffusion/src/families/ltx/embeddings.test.ts
```

Result: `20 pass`, `0 fail` for the focused LTX tests.

## Independent Review

Singer reviewed the LTX-2 next tranche before implementation. The review
identified prepared dual-latent denoising as the smallest product-correct slice
after latent upsampling, with explicit attention to x0-domain CFG, shared
video/audio timesteps, coordinate duplication for CFG, and rejection of
unsupported STG/modality/rescale paths.

Singer also reviewed the implementation before commit. That pass caught an
initial divergence where video and audio schedules could be resolved
independently; the final implementation uses one shared schedule/timestep path
for both modalities, matching Diffusers.

## Out-of-scope Drift Noticed

No additional drift was changed in this tranche.

## Remaining Risks / Follow-ups

- Full LTX-2 checkpoint generation still needs text connector, transformer,
  video/audio decoder, vocoder, and proof CLI tranches.
- The prepared loop has synthetic shape and CFG coverage, not a real checkpoint
  audio-video proof.
