# Qwen-Image Edit Snapshot Config Review

## Summary

This tranche adds explicit Diffusers snapshot/config recognition for
`QwenImageEditPipeline` and `QwenImageEditPlusPipeline`. The package now
distinguishes base Qwen-Image text-to-image snapshots from Qwen-Image Edit and
Edit Plus manifests, recognizes the required `Qwen2VLProcessor` component, and
loads shared Qwen-Image transformer/VAE configs for edit snapshots without
claiming edit generation.

The runtime fence remains deliberate: `Qwen/Qwen-Image-Edit-2511` preserves
`zero_cond_t: true`, and `QwenImageTransformer2DModel` still rejects that branch
until its execution semantics are implemented.

## Files Reviewed

- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/snapshot-manifest.ts`
- `packages/diffusion/src/pretrained/snapshot-file-selection.ts`
- `packages/diffusion/src/pretrained/model-index.test.ts`
- `packages/diffusion/src/pretrained/snapshot-file-selection.test.ts`
- `packages/diffusion/src/families/qwen-image/config.ts`
- `packages/diffusion/src/families/qwen-image/config.test.ts`
- `packages/diffusion/README.md`
- `PLAN.md`
- `docs/gates-and-milestones.md`
- `continuity.md`
- `MEMORY.md`

## Reference Audit

- `Qwen/Qwen-Image-Edit-2511` declares
  `_class_name: "QwenImageEditPlusPipeline"` and uses `processor:
  ["transformers", "Qwen2VLProcessor"]` alongside scheduler, text encoder,
  tokenizer, transformer, and VAE components:
  https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/main/model_index.json
- The 2511 model card uses `QwenImageEditPlusPipeline` with one or more input
  images and names its product improvements as consistency, built-in LoRA
  effects, industrial design, and geometric reasoning:
  https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Diffusers documents Qwen Image Edit Plus as the multi-reference image-edit
  pipeline with `Qwen2VLProcessor`, `Qwen2_5_VLForConditionalGeneration`,
  `QwenImageTransformer2DModel`, and `AutoencoderKLQwenImage`:
  https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- The original Qwen-Image Edit manifest declares
  `_class_name: "QwenImageEditPipeline"` with the same processor/component
  boundary:
  https://huggingface.co/Qwen/Qwen-Image-Edit/blame/refs%2Fpr%2F5/model_index.json

## Tensor Lifetime Audit

No tensor-producing runtime path changed. This tranche changes JSON manifest
parsing, remote file selection, and component metadata inspection only.

## Memory / Performance Evidence

No generation, VAE encode/decode, denoising, scheduler stepping, tokenizer,
text-encoder, or cache code changed. The Qwen-Image text-to-image proof command
still accepts only `qwen-image` manifests, and edit snapshots remain out of
the executable proof path.

## Evidence

- `bun test packages/diffusion/src/pretrained/model-index.test.ts`
- `bun test packages/diffusion/src/pretrained/snapshot-file-selection.test.ts`
- `bun test packages/diffusion/src/families/qwen-image/config.test.ts`

## Independent Review

Dewey the 2nd performed a read-only second pass and recommended this as the
next low-risk Phase 10 tranche as long as it stayed snapshot/config recognition
only. The review called out the `processor` component, 2511's
`QwenImageEditPlusPipeline` manifest class, and `zero_cond_t: true` as the
important upstream facts to preserve.

## Remaining Risks / Follow-ups

- Edit generation still needs reference-image processor conditioning,
  package-owned image VAE encode, reference latent concatenation, edit-specific
  prompt template handling, `zero_cond_t` execution, and multi-reference shape
  tests.
- Qwen-Image img2img, inpaint, ControlNet, layered, LoRA, quantized sidecars,
  and real edit checkpoint evidence remain separate Phase 10 tranches.
