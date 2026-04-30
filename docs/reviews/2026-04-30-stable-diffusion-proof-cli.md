# Stable Diffusion Proof CLI

## Summary

Added the finite `examples/stable-diffusion/index.ts` proof command. The command
loads a local Diffusers Stable Diffusion or SDXL snapshot, composes the
example-owned prompt conditioner with the package-owned diffusion pipeline,
samples one image under the shared runtime lock, and writes an uncompressed BMP
artifact from the returned NHWC image tensor.

The tranche keeps `@mlxts/diffusion` tensor-only. Host image output and
cross-package CLIP/tokenizer composition stay in the example boundary.

## Files Reviewed

- `examples/stable-diffusion/index.ts`
- `examples/stable-diffusion/image-output.ts`

## Reference Audit

- `.reference/mlx-examples/stable_diffusion/stable_diffusion/__init__.py`
  confirms the application-layer shape: text conditioning feeds a sampler, and
  generated image tensors are converted to host-visible image artifacts outside
  the model modules.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  and `pipeline_stable_diffusion_xl.py` keep prompt encoding, scheduler
  denoising, VAE decode, and output conversion as distinct stages. This command
  follows that stage split without copying Diffusers' Python pipeline object.
- `examples/qwen3_5-image/index.ts` is the local AXI reference for parse-before
  lock, progress-on-stderr, structured stdout, and stable `0` / `1` / `2` exits.

## Tensor Lifetime Audit

The CLI uses `using` for the loaded diffusion bundle, prompt conditioner,
returned prompt-conditioning tensors, RNG key, and generated image tensor. The
pipeline keeps per-step evaluation at its package default and the command does
not override it. BMP writing calls `image.eval()` once before copying to host
with `toTypedArray()`, then writes host bytes only.

## Memory / Performance Evidence

This tranche adds a finite proof command and does not change denoising math,
scheduler behavior, VAE decode, or transformer CLIP execution. It makes no
quality or throughput claim. The default validation gate covers command shape,
structured output, prompt-conditioning composition, and deterministic BMP
encoding without running heavy checkpoint generation.

Real checkpoint image quality/parity evidence remains a separate run because
this machine did not have a cached Stable Diffusion Diffusers snapshot at commit
time.

## Independent Review

Anscombe completed a read-only second pass over the existing diffusion and
example surfaces. The review recommended keeping the CLI and image artifact
writer entirely under `examples/stable-diffusion`, using the existing
`StableDiffusionPipelineBundle.generateImage()` surface, and choosing BMP for
the first artifact format to avoid adding host image dependencies.

## Validation

- `bun run check:phase10-proofs`

## Remaining Risks / Follow-ups

- A real cached Stable Diffusion or SDXL checkpoint proof still needs to be run
  and recorded before claiming image quality or reference parity.
- PNG/JPEG output, Hub-backed Diffusers snapshot resolution, image-to-image,
  inpainting, and safety-checker semantics remain separate Phase 10 tranches.
