# Diffusion SDXL Real Checkpoint Proof

## Summary

Official SDXL fp16 Hub snapshots now load through the Stable Diffusion proof
command and run a bounded image generation proof. The tranche tightened Hub
cache manifest handling, SDXL model-index parsing, and two UNet construction
details that blocked real checkpoint parity with Diffusers.

## Files Reviewed

- `packages/diffusion/src/pretrained/snapshot-manifest.ts`
- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/families/stable-diffusion/unet.ts`
- `packages/diffusion/src/families/stable-diffusion/unet-transformer.ts`

## Reference Audit

Hugging Face Hub snapshots store downloaded files as symlinks into the local
blob cache. Component-local safetensors therefore need file-target checks
rather than `Dirent.isFile()` checks.

`stabilityai/stable-diffusion-xl-base-1.0` publishes `add_watermarker` in
`model_index.json` as scalar pipeline metadata. The loader preserves it as
pipeline config instead of treating it as a component; this tranche does not
implement watermarking or claim watermark parity.

Diffusers `UNet2DConditionModel` builds up blocks with the previous up block's
output channels after the first up block. SDXL checkpoint tensors require that
previous-output channel convention for `up_blocks.*.resnets.*.conv1.weight`.

Diffusers attention projections use biasless query/key/value projections and a
biased output projection. SDXL UNet safetensors contain `to_q.weight`,
`to_k.weight`, and `to_v.weight` without matching bias tensors, while
`to_out.0.bias` is present.

## Tensor Lifetime Audit

The manifest and model-index changes are host-side file/config handling only.
The UNet construction changes alter module parameter shapes and bias ownership
before model execution. They do not add tensor-producing forward expressions,
change `MxArray` disposal structure, or hide tensor lifetimes inside helpers.

## Memory / Performance Evidence

Real checkpoint proof command:

```bash
bun run examples/stable-diffusion/index.ts stabilityai/stable-diffusion-xl-base-1.0 --variant fp16 --cache-dir .tmp/hf-diffusion-proof-cache --prompt "a small red apple on a white table, product photo" --output .tmp/stable-diffusion/sdxl-base-proof.bmp --steps 2 --height 256 --width 256 --guidance-scale 5 --seed 7 --dtype float16 --json
```

The resolver used the cached official fp16 snapshot at revision
`462165984030d82259a11f4367a4eed129e94a7b`, selected `21` files totaling
`6,941,189,357` bytes, loaded the Stable Diffusion XL pipeline, encoded prompt
conditioning, ran two denoise steps, and wrote
`.tmp/stable-diffusion/sdxl-base-proof.bmp` with `196,662` bytes.

This is a product proof of SDXL checkpoint loading and bounded image execution,
not a performance optimization claim. No throughput or quality benchmark claim
is made in this tranche.

## Independent Review

Dirac reviewed the diff for checkpoint correctness, tensor-lifetime concerns,
and out-of-scope drift. The review found no blocking issue, confirmed the
SSD-1B drift belongs out of scope, and requested explicit tests for symlinked
weight indexes plus cross-attention projection bias layout. Those tests are
included in this tranche.

## Validation

- `bun test packages/diffusion/src/families/stable-diffusion/unet.test.ts`
- `bun test packages/diffusion/src/pretrained/model-index.test.ts`
- `bun test packages/diffusion/src/pretrained/model-index.test.ts packages/diffusion/src/families/stable-diffusion/unet.test.ts packages/diffusion/src/families/stable-diffusion/weights.test.ts`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- Real SDXL proof command above
- `bun run validate`

## Out-of-scope Drift Noticed

`segmind/SSD-1B` was useful as a remote snapshot selection probe, but it is not
covered by this SDXL proof. Its UNet config uses residual-only
`mid_block_type: "UNetMidBlock2D"` and nested `transformer_layers_per_block`
shapes that need a separate architecture tranche before it is claimed.

## Remaining Risks / Follow-ups

- FLUX.1 still needs its own real checkpoint image proof after the SDXL baseline.
- Existing polluted Hub cache directories can still contain previously
  downloaded unselected files; clean cache directories prove resolver selection.
