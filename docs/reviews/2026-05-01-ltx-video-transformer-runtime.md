# Runtime Review: LTX Video Transformer Runtime

## Summary

This tranche adds the classic LTX-Video transformer execution surface for
packed video-token denoising.

`LtxVideoTransformer3DModel` implements the prepared `LtxVideoDenoiser`
contract: it consumes `[batch, tokens, inChannels]` packed latents plus prepared
caption embeddings, semantic prompt masks, raw FlowMatch timesteps, and video
geometry, then returns `[batch, tokens, outChannels]` velocity predictions. It
also adds Diffusers transformer weight-name mapping/loading so the next tranche
can load official LTX transformer snapshots instead of stopping at synthetic
runtime coverage.

This does not add the LTX VAE, latent denormalization, latent upsampling, text
encoder orchestration, video artifact writing, LTX-2 audio/video denoising, or
proof commands.

## Files Reviewed

- `packages/diffusion/src/families/ltx/attention.ts`
- `packages/diffusion/src/families/ltx/blocks.ts`
- `packages/diffusion/src/families/ltx/conditioning.ts`
- `packages/diffusion/src/families/ltx/tensor-utils.ts`
- `packages/diffusion/src/families/ltx/transformer.ts`
- `packages/diffusion/src/families/ltx/transformer-weights.ts`
- `packages/diffusion/src/families/ltx/transformer.test.ts`
- `packages/diffusion/src/families/ltx/transformer-weights.test.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

Transformer forward retains only the current hidden-state loop value, frees each
previous value after block replacement, and frees final AdaLN modulation tensors
after output projection. The RoPE cache is model-owned and disposed from
`[Symbol.dispose]`; it is keyed by batch, latent video geometry, and RoPE
interpolation scale so denoising steps do not rebuild host-generated frequency
tensors.

Attention projection temporaries are owned inside `try/finally` blocks. RoPE
application upcasts q/k tensors to fp32 for rotation and casts the result back
to the query dtype before attention. Semantic prompt masks are converted once
per forward to a boolean `[batch, 1, 1, textLength]` SDPA mask and freed before
return or error.

Safetensor loading follows existing diffusion shard-iterator ownership:
unexpected tensors are freed immediately, transformed tensors replace existing
parameters only after shape checks, and assigned tensors are not double-freed.

No native bindings, FFI symbol declarations, or tensor-producing primitive
lists changed.

## Reference Parity

The implementation follows current Diffusers
`.reference/diffusers/src/diffusers/models/transformers/transformer_ltx.py`:

- `proj_in` precedes timestep and caption conditioning
- `AdaLayerNormSingle(use_additional_conditions=false)` produces block
  modulation plus final embedded timestep
- caption embeddings use PixArt GELU-tanh projection into transformer hidden
  size
- block self-attention uses RMS-normalized Q/K plus classic LTX RoPE before
  head reshape
- block cross-attention consumes projected caption tokens and semantic prompt
  masks without RoPE
- final affine-free LayerNorm uses the model-level two-row scale/shift table
  before `proj_out`

The first runtime deliberately rejects non-classic branches: transformer
`patchSize` or `patchSizeT` other than 1, affine block RMSNorm weights,
`qk_norm` values other than `rms_norm_across_heads`, activation functions other
than `gelu-approximate`, and `crossAttentionDim !== hiddenSize`.

## Memory / Performance Evidence

No performance claim is made. The main performance-sensitive choice is that
classic RoPE tensors are cached at model scope instead of regenerated inside
each denoising step.

Focused gates passed:

```bash
bun test packages/diffusion/src/families/ltx
bun run typecheck
bun run check:file-lines
bun run validate
```

The full repo gate passed after adding focused LTX loader and shared-config
coverage for malformed shard manifests and config helper variants.

## Independent Review

Godel ran a read-only reference review before implementation. The review
recommended a classic-only packed-sequence denoiser, strict semantic prompt
mask handling, early rejection of non-classic variants, Diffusers-compatible
weight-name mapping, and synthetic tiny-runtime tests. This patch follows that
scope and does not add LTX-2, VAE runtime, custom `video_coords`, cache/LoRA
support, or proof-command behavior.

## Remaining Risks / Follow-ups

The next LTX tranche is VAE decode/denormalization plus latent upsampling, then
a finite `examples/ltx-video` proof command. LTX-2 remains separate because its
current Diffusers architecture has video and audio latent streams plus connector
and vocoder components rather than the single classic video-token path added
here.
