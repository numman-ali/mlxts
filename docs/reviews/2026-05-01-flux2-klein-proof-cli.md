# FLUX.2 Klein Proof CLI

## Summary

`examples/flux2` now owns the application-layer proof command for FLUX.2 Klein
text-to-image generation. The example resolves local or Hub Diffusers
`Flux2KleinPipeline` snapshots, loads the package-owned FLUX.2 transformer, VAE,
and FlowMatch scheduler, composes Qwen3 prompt conditioning from the checkpoint
tokenizer/chat template/text encoder, runs denoising, and writes one BMP
artifact.

This tranche does not add reference-image conditioning, KV-cache pipeline
variants, inpainting, editing, LoRA, GGUF/single-file loading, quantized
sidecars, or bounded real checkpoint evidence.

## Files Reviewed

- `examples/flux2/index.ts`
- `examples/flux2/conditioning.ts`
- `examples/flux2/conditioning-runtime.ts`
- `examples/flux2/conditioning-result.ts`
- `examples/flux2/conditioning-types.ts`
- `examples/flux2/image-output.ts`
- `examples/image-proof/verify-report.ts`

## Reference Check

- `.reference/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`
  was reviewed for Qwen3 prompt conditioning: one user message, chat template
  generation prompt enabled, Qwen thinking disabled, default hidden-state layers
  `9`, `18`, and `27`, default guidance scale `4`, and empty-string negative
  prompt for non-distilled classifier-free guidance.
- `.reference/diffusers/src/diffusers/models/transformers/transformer_flux2.py`
  and the existing `@mlxts/diffusion` FLUX.2 runtime were reviewed for the
  prepared-embedding boundary consumed by the denoiser.

## Tensor Lifetime Audit

Prompt-conditioner inputs, Qwen3 hidden-state outputs, selected prompt embeds,
negative prompt embeds, RNG keys, generated images, transformer, VAE, and text
encoder handles all stay behind explicit `using` scopes or
`[Symbol.dispose]()` results. `Flux2KleinPromptConditioningResult` owns and
frees prompt embedding tensors that cross from the example into
`@mlxts/diffusion`.

The image output writer evaluates and copies the final NHWC image tensor without
taking ownership of the caller-owned tensor.

## Memory / Performance Evidence

This is a finite proof-command tranche, not a throughput or image-quality
benchmark. Fixture-backed tests cover prompt-conditioning shape and ownership,
AXI stdout/stderr separation, usage/runtime error behavior, BMP output, and
saved-report verification.

Focused evidence:

```bash
bun test examples/flux2
bun test examples/image-proof
bunx tsc -p tsconfig.phase10-examples.json --pretty false
```

## Independent Review

Russell the 2nd reviewed the intended FLUX.2 Klein proof boundary before the
final implementation. The review confirmed that the example must not reuse
Z-Image's penultimate hidden-state path or Qwen-Image's Qwen2.5-VL wrapper, and
called out the Diffusers Qwen3 conditioning recipe of hidden states `9/18/27`
with thinking disabled.

## Remaining Risks / Follow-ups

- The first real `black-forest-labs/FLUX.2-klein-4B` checkpoint proof still
  needs to run and record artifact evidence.
- The Qwen3 text-encoder path pads token ids when a pad token exists but does
  not yet pass a Diffusers-style attention mask into `LlamaLikeModel`.
  Generated proof commands are therefore product-capability evidence, not exact
  Diffusers text-conditioning parity.
- Reference-image, KV-cache, inpaint, editing, LoRA, GGUF/single-file, and
  quantized FLUX.2 variants remain separate tranches.

## Out-of-scope Drift Noticed

- `.reference/transformers` remains in an existing unresolved merge state and
  was not refreshed during this tranche.
