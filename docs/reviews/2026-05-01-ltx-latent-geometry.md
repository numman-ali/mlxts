# Runtime Review: LTX Latent Geometry

## Summary

This tranche adds package-owned latent geometry and packing helpers for the
already-recognized LTX-Video and LTX-2 Diffusers snapshot families.
`@mlxts/diffusion` now derives LTX video latent shapes, packs/unpacks BCFHW
video latents into Diffusers-compatible token sequences, derives LTX-2 audio
latent duration and mel geometry, and packs/unpacks BCLM audio latents into the
current Diffusers token order.

This does not add LTX transformer execution, VAE execution, latent upsampling,
artifact writing, or proof commands.

## Files Reviewed

- `packages/diffusion/src/families/ltx/latents.ts`
- `packages/diffusion/src/families/ltx/latents.test.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

The new tensor helpers keep disposable reshaped/transposed intermediates visible
with `using` declarations before returning the retained reshape result. Scheduler
initial-noise helpers create a sampled latent, pack it while the sampled source
is still live, and dispose the source before returning the packed tensor.

No nested tensor-producing expressions hide intermediate ownership. No module
fields, model forward paths, or safetensor weight ownership paths changed.

## Reference Parity

LTX video packing mirrors Diffusers pipeline tokenization:

- `.reference/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`

LTX-2 audio packing mirrors `prepare_audio_latents` and `_pack_audio_latents`
from the LTX-2 pipeline. Audio latent length uses Python-style half-even rounding
so `.5` ties match Diffusers `round(...)` behavior instead of JavaScript
`Math.round(...)`.

Video frame shape uses Diffusers floor-style latent-frame derivation:
`floor((numFrames - 1) / temporalCompressionRatio) + 1`.

## Memory / Performance Evidence

No generation hot path changed, and no performance claim is made. The focused
LTX gate passed:

```bash
bun test packages/diffusion/src/families/ltx
```

Result: 15 tests passed, covering LTX-Video and LTX-2 config parsing, video
latent shape derivation, video pack/unpack ordering, scheduler-created video
latents, LTX-2 audio length rounding, audio pack/unpack ordering, optional audio
patch packing, scheduler-created audio latents, and malformed packed-shape
rejections.

## Independent Review

Boole performed a read-only second-opinion review against local Diffusers LTX
references. The review caught two parity issues before commit: JavaScript
rounding did not match Python half-even `round(...)` for LTX-2 audio length,
and video latent frame derivation was stricter than Diffusers for non-`8n+1`
frame requests. Both issues are fixed in the landed helper and covered by tests.

The review also confirmed that `createLtx2AudioInitialLatents` matches current
Diffusers behavior when audio patch sizes are absent. The helper now accepts
optional audio patch sizes so future non-`1x1` audio patch configs do not need a
new API shape.

## Remaining Risks / Follow-ups

LTX transformer execution, video VAE execution, LTX-2 audio VAE/vocoder
execution, latent upsampling, artifact encoding, and AXI proof commands remain
future Phase 10 tranches.

These helpers describe pipeline token packing. They must not be reused for the
different internal patchification used inside the LTX video and audio VAEs.
