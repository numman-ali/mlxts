# Qwen-Image Proof CLI Review

## Summary

`examples/qwen-image` now owns the application-layer proof command for base
Diffusers `QwenImagePipeline` snapshots. The command resolves local or Hub
snapshots, loads the package-owned Qwen-Image transformer, VAE, and FlowMatch
Euler scheduler, encodes prompts through Qwen2.5-VL using the Diffusers
Qwen-Image prompt template, runs denoising, and writes one BMP artifact.

The tranche does not add Qwen-Image edit, control, inpaint, img2img, LoRA, or
guidance-distilled model variants.

## Files Reviewed

- `examples/qwen-image/index.ts`
- `examples/qwen-image/conditioning.ts`
- `examples/qwen-image/conditioning-runtime.ts`
- `examples/qwen-image/conditioning-result.ts`
- `examples/qwen-image/conditioning-types.ts`
- `examples/qwen-image/image-output.ts`
- `tsconfig.phase10-examples.json`
- `package.json`

## Tensor Lifetime Audit

Prompt conditioning owns the Qwen2.5-VL text encoder output only long enough to
retain the final hidden-state slice after the 34-token Diffusers template drop.
The original text output and input token tensor are freed in `finally` blocks.
The disposable conditioning result owns positive and optional negative prompt
embedding tensors. The image output writer evaluates and copies the final NHWC
image tensor without taking tensor ownership.

## Memory / Performance Evidence

No performance optimization claim is made. The proof command keeps the same
bounded single-image command shape as the Stable Diffusion, FLUX, and Z-Image
examples and acquires the shared runtime command lock before loading model
weights.

Focused validation:

- `bun test examples/qwen-image`
- `tsc -p tsconfig.phase10-examples.json`

## Independent Review

Schrodinger the 2nd reviewed the Qwen-Image proof boundary before implementation
and confirmed the example-owned layout, Diffusers fixed prompt wrapper, final
hidden-state conditioning, 34-token drop, and true-CFG negative-conditioning
rules.

## Remaining Risks / Follow-ups

- Official `Qwen/Qwen-Image-2512` real checkpoint proof evidence still needs a
  bounded local run after this CLI tranche.
- The proof command stays batch-1. Batched prompts need text attention-mask
  semantics before padded Qwen2.5-VL conditioning is honest.
- Qwen-Image edit/control/inpaint/img2img variants remain separate product
  tranches.
