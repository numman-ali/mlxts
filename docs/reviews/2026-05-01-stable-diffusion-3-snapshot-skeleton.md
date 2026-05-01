# Runtime Review: Stable Diffusion 3 snapshot skeleton

## Summary

Stable Diffusion 3 / 3.5 now has a bounded `@mlxts/diffusion`
snapshot/config skeleton. The package recognizes Diffusers
`StableDiffusion3Pipeline` manifests, records the triple text-encoder/tokenizer
layout, and parses `SD3Transformer2DModel` plus AutoencoderKL configs without
claiming runtime generation or checkpoint proof.

SD3.5 stays on the same family surface through `dual_attention_layers` and
`qk_norm: "rms_norm"` metadata. Img2img, inpaint, ControlNet, IP-Adapter,
LoRA, prompt encoding, transformer execution, weight loading, and real
checkpoint proof remain explicit follow-ups.

## Files Reviewed

- `packages/diffusion/src/pretrained/model-index.ts`
- `packages/diffusion/src/pretrained/pipeline-specs.ts`
- `packages/diffusion/src/pretrained/model-index.test.ts`
- `packages/diffusion/src/families/stable-diffusion-3/config.ts`
- `packages/diffusion/src/families/stable-diffusion-3/config.test.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- Hugging Face Diffusers documents `StableDiffusion3Pipeline` as
  `SD3Transformer2DModel`, `FlowMatchEulerDiscreteScheduler`, `AutoencoderKL`,
  two `CLIPTextModelWithProjection` encoders, and one `T5EncoderModel`:
  https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3
- Local Diffusers source confirms the same registered modules plus optional
  SigLIP IP-Adapter components:
  `.reference/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`
- Local Diffusers `SD3Transformer2DModel` source identifies learned patch
  positional embeddings, `caption_projection_dim`, `pooled_projection_dim`,
  `dual_attention_layers`, and optional `qk_norm`:
  `.reference/diffusers/src/diffusers/models/transformers/transformer_sd3.py`
- `hf-internal-testing/tiny-sd3-pipe` provides public tiny manifest/config
  fixtures with the same component layout.
- `stabilityai/stable-diffusion-3.5-large`,
  `stabilityai/stable-diffusion-3.5-large-turbo`, and
  `stabilityai/stable-diffusion-3-medium-diffusers` advertise
  `diffusers:StableDiffusion3Pipeline` and are gated, so authenticated proof is
  a later step.

## Tensor Lifetime Audit

This tranche adds manifest and JSON config parsing only. It constructs no MLX
tensors, loads no safetensors, and adds no tensor-producing runtime code.

## Memory / Performance Evidence

No performance claim is made. The focused validation only proves manifest and
config acceptance/rejection behavior:

```bash
bun test packages/diffusion/src/pretrained/model-index.test.ts
bun test packages/diffusion/src/families/stable-diffusion-3/config.test.ts
bunx tsc -p packages/diffusion/tsconfig.json --pretty false
```

## Independent Review

Lovelace the 2nd performed a read-only SD3 skeleton audit. The review
recommended landing only `StableDiffusion3Pipeline` snapshot/config recognition,
including the triple text stack and SD3.5 dual-attention fields, and leaving
runtime, weight mapping, prompt encoding, img2img/inpaint/control/IP-Adapter,
and gated proof out of this tranche.

## Remaining Risks / Follow-ups

- Runtime tensor execution and MMDiT weight mapping/loading are not implemented.
- CLIP/T5 prompt encoding stays outside `@mlxts/diffusion` and needs a proof
  command when runtime lands.
- Official Stability SD3/3.5 checkpoints require authenticated access before a
  real checkpoint proof can be recorded.
- SD3 img2img, inpaint, ControlNet, PAG, IP-Adapter, and LoRA remain separate
  product tranches.
