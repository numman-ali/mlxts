# Runtime Review: LTX RoPE Geometry

## Summary

This tranche adds package-owned coordinate and RoPE geometry helpers for the
already-recognized LTX-Video and LTX-2 Diffusers families.

Classic LTX now has helpers for Diffusers-compatible video coordinate grids and
interleaved cosine/sine frequency tensors. LTX-2 now has helpers for video
patch-boundary coordinates, audio patch-boundary coordinates, and interleaved
or split cosine/sine frequency tensors.

This does not add LTX transformer execution, attention execution, VAE execution,
LTX-2 cross-modality attention, latent upsampling, artifact writing, or proof
commands.

## Files Reviewed

- `packages/diffusion/src/families/ltx/embeddings.ts`
- `packages/diffusion/src/families/ltx/embeddings-rope.ts`
- `packages/diffusion/src/families/ltx/embeddings.test.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

The new helpers create coordinate and frequency tensors from owned typed arrays
and return fresh `MxArray` handles. No temporary MLX tensor intermediates are
hidden inside nested expressions. Callers own returned cosine/sine arrays
individually, and tests free both arrays through one helper in every path.

The LTX-2 coordinate-to-RoPE helper performs a host readback of the coordinate
tensor, so it is explicitly treated as precompute scaffolding. Future transformer
runtime work must compute or cache the returned cosine/sine tensors outside the
denoising loop, or replace the conversion with MLX tensor ops before using it
as a hot-path forward primitive.

No module fields, model forward paths, safetensor weight ownership paths, or
scheduler state changed.

## Reference Parity

Classic LTX coordinate scaling and interleaved RoPE follow:

- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx.py`

LTX-2 video/audio patch-boundary coordinate generation and interleaved/split
RoPE tensor layouts follow:

- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx2.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`

The focused tests cover token order, temporal causal offset handling, video FPS
conversion, audio mel-to-second conversion, classic LTX padding semantics,
LTX-2 interleaved output shape and value order, LTX-2 split attention-head
output shape and front-padding order, and malformed runtime `ropeType` values.

## Memory / Performance Evidence

No generation hot path changed, and no performance claim is made. These helpers
are request-geometry scaffolding for future LTX runtime work.

Focused gates passed:

```bash
bun test packages/diffusion/src/families/ltx
bun run lint
bun run check:file-lines
bun run typecheck
bun run check:runtime-review
bun run validate
```

## Independent Review

Boole provided the pre-implementation next-tranche recommendation and then ran
a read-only review of this tranche against the local Diffusers LTX references.
That review caught four tightenings before commit: classic LTX patch-size
coordinates must not step the grid by patch size, malformed runtime `ropeType`
values must throw instead of falling through to interleaved mode, host readback
must stay documented as precompute-only scaffolding, and the RoPE tests needed
nontrivial interleaved/split values. All four are integrated in the landed
change.

## Remaining Risks / Follow-ups

LTX-Video packed-latent denoising is the next bounded runtime foundation. Full
LTX transformer execution, VAE execution, LTX-2 audio/video denoising,
cross-modality attention, latent upsampling, output encoding, and proof
commands remain future Phase 10 tranches.
