# Qwen-Image Edit Transformer Segment Review

## Summary

This tranche lands the transformer-side semantics required before Qwen-Image
Edit/Edit Plus denoising can be wired honestly. The Qwen-Image transformer now
accepts ordered target/reference RoPE image segments, validates that the
concatenated hidden-state sequence matches those segments, and supports
Diffusers `zero_cond_t` image modulation by applying the real timestep to target
tokens and a zero timestep to reference tokens.

Base text-to-image calls still pass a single `imageShape`; edit denoising
concat/slice and processor reference-image conditioning remain separate
tranches.

## Files Reviewed

- `packages/diffusion/src/families/qwen-image/blocks.ts`
- `packages/diffusion/src/families/qwen-image/embeddings.ts`
- `packages/diffusion/src/families/qwen-image/latents.ts`
- `packages/diffusion/src/families/qwen-image/transformer.ts`
- `packages/diffusion/src/families/qwen-image/embeddings.test.ts`
- `packages/diffusion/src/families/qwen-image/transformer.test.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

Local Diffusers reference commit:

```bash
git -C .reference/diffusers log -1 --date=short --pretty='%h %cd %s'
```

Result: `c8eba433a 2026-05-01 [agents docs] update models.md with class attributes and attention mask (#13665)`.

Reviewed reference points:

- `.reference/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py`
- `.reference/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py`
- `.reference/diffusers/src/diffusers/modular_pipelines/qwenimage/before_denoise.py`
- `.reference/diffusers/src/diffusers/modular_pipelines/qwenimage/denoise.py`

The shipped TypeScript path matches the relevant prerequisites: edit pipelines
pass target and reference latent shapes, the transformer builds RoPE over all
image segments, and `zero_cond_t` doubles timestep embeddings only for image
modulation while text and final output normalization use the real timestep half.

## Tensor Lifetime Audit

The new tensor-producing branches keep intermediates visible. Multi-segment
RoPE construction retains each segment in a local list and frees every segment
after concatenation. `zero_cond_t` creates the concatenated timestep tensor once
per forward call, creates the modulation index once per forward call, and frees
the index in the existing transformer `finally` block. Indexed modulation uses
visible slices, selectors, and `where` outputs without hiding disposable tensor
creation inside nested calls.

Base non-edit calls do not allocate a modulation index and continue through the
single-shape RoPE path.

## Memory / Performance Evidence

Focused tests:

```bash
bun test packages/diffusion/src/families/qwen-image/embeddings.test.ts packages/diffusion/src/families/qwen-image/transformer.test.ts packages/diffusion/src/families/qwen-image/pipeline.test.ts packages/diffusion/src/families/qwen-image/conditioning.test.ts
```

Result: 22 pass, 0 fail.

Focused gate:

```bash
bun run typecheck
```

Result: passed.

Full validation:

```bash
bun run validate
```

Result: passed.

This is a runtime semantics change, not a performance optimization. The base
text-to-image path remains one segment and avoids the `zero_cond_t` modulation
index allocation unless the checkpoint config requires it.

## Independent Review

Herschel the 2nd performed a read-only second pass over the local Qwen-Image
runtime and Diffusers references. The review recommended transformer semantic
support first, specifically multi-segment RoPE plus `zero_cond_t`, and advised
leaving edit denoise concat/slice for the next tranche so the pipeline does not
fake-test against the wrong geometry.

## Remaining Risks / Follow-ups

- Qwen-Image Edit/Edit Plus still needs processor-owned reference-image prompt
  conditioning with the Qwen2VL image template.
- The next tranche should add denoising concat/slice: pass target plus reference
  latents to the transformer, then slice predictions back to the target length
  before scheduler stepping.
- Real edit checkpoint evidence remains blocked until the processor and
  denoising loop are wired together.
