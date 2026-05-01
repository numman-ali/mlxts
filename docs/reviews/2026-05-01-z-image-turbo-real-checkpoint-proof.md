# Runtime Review: Z-Image Turbo Real Checkpoint Proof

## Summary

Official `Tongyi-MAI/Z-Image-Turbo` now has a bounded real checkpoint proof
through `examples/z-image`. The proof resolved the current Diffusers
`ZImagePipeline` snapshot, loaded the Qwen3 text encoder, Z-Image transformer,
FlowMatch Euler scheduler, and AutoencoderKL VAE, encoded prompt conditioning,
ran two denoising steps, and wrote a 256x256 BMP artifact.

This is a product capability proof, not a throughput or image-quality benchmark.

## Files Reviewed

- `packages/diffusion/src/families/z-image/latents.ts`
- `packages/diffusion/src/families/z-image/pipeline.ts`
- `packages/diffusion/src/families/z-image/transformer.ts`
- `packages/diffusion/src/families/z-image/weights.ts`
- `packages/diffusion/src/families/flux/autoencoder.ts`
- `packages/diffusion/src/pretrained/snapshot-source.ts`
- `examples/z-image/index.ts`
- `examples/z-image/conditioning.ts`
- `examples/z-image/conditioning-runtime.ts`
- `examples/z-image/image-output.ts`

## Snapshot Evidence

The resolved snapshot is:

```text
Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6
```

`model_index.json` declares:

- pipeline: `ZImagePipeline`
- scheduler: `FlowMatchEulerDiscreteScheduler`
- text encoder: `Qwen3Model`
- tokenizer: `Qwen2Tokenizer`
- transformer: `ZImageTransformer2DModel`
- VAE: `AutoencoderKL`

The selected Diffusers snapshot contained 19 files and
`32,848,305,533` bytes. The transformer config used `dim=3840`,
`n_layers=30`, `n_refiner_layers=2`, `n_heads=30`, `cap_feat_dim=2560`,
`axes_dims=[32,48,48]`, and `in_channels=16`.

## Tensor Lifetime Audit

No production runtime code changed in this proof tranche. The proof exercised the
existing runtime path that keeps Z-Image latents, padded feature sequences,
denoiser inputs, prompt conditioning, and generated image tensors behind
explicit `using` or paired disposal boundaries. The proof command holds the
shared runtime lock for the whole model-load and denoising run.

## Validation

Focused static proof gate:

```bash
bun run check:phase10-proofs
```

Result: 61 pass, 0 fail.

Real checkpoint proof command:

```bash
bun run examples/z-image/index.ts Tongyi-MAI/Z-Image-Turbo \
  --cache-dir .tmp/hf-diffusion-proof-cache \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/z-image/z-image-turbo-official-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --guidance-scale 0 \
  --seed 7 \
  --dtype bfloat16 \
  --json
```

Result:

- resolved revision:
  `f332072aa78be7aecdf3ee76d5c247082da564a6`
- output: `.tmp/z-image/z-image-turbo-official-proof.bmp`
- output bytes: `196,662`
- image size: `256x256`
- prompt truncated: `false`
- elapsed: `843,956.6 ms`, including initial Hub download

The BMP artifact was verified as a `256 x 256 x 24` Windows BMP. A temporary PNG
conversion opened successfully for visual inspection.

## Independent Review

Hooke performed a read-only product/architecture pass across Phase 10 image
support and recommended the Z-Image-Turbo real checkpoint proof as the next
non-speculative tranche. The reasoning was that Z-Image already had the runtime
and proof CLI, while the official checkpoint evidence was the remaining product
claim gap.

## Remaining Risks / Follow-ups

- This proof is bounded to `256x256`, two denoising steps, batch size 1, and
  `guidance_scale=0`.
- It is not a throughput benchmark and does not claim 1024px/default-step image
  quality.
- CFG, ControlNet, image-to-image, inpainting, Omni/SigLIP, multi-batch, and
  mflux quantized sidecars remain unsupported until their runtime semantics are
  designed deliberately.
- Qwen-Image / Qwen-Image-2512 still needs its own official full checkpoint
  proof.
