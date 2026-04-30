# Stable Diffusion Prompt Conditioning

## Summary

Added `examples/stable-diffusion` as the application-layer Stable Diffusion
prompt-conditioning workbook. The example composes `@mlxts/diffusion`,
`@mlxts/transformers`, and `@mlxts/tokenizers` without changing the package
dependency graph: diffusion still consumes tensors only, transformers still owns
CLIP encoders, and tokenizers still owns CLIP vocab/merges.

The conditioner supports SD 1.x / SD 2.x last-hidden-state conditioning and SDXL
dual-CLIP conditioning with penultimate hidden states, projected text embeddings,
and Diffusers-style added time ids.

## Files Reviewed

- `examples/stable-diffusion/conditioning-runtime.ts`
- `examples/stable-diffusion/conditioning-result.ts`
- `examples/stable-diffusion/conditioning-types.ts`
- `examples/stable-diffusion/conditioning.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  uses CLIP tokenizer padding/truncation to max length, runs the CLIP text
  encoder, and passes `last_hidden_state` into the UNet cross-attention path.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`
  uses `hidden_states[-2]` from both CLIP encoders, concatenates them along the
  hidden axis, uses projected pooled text embeddings from `text_encoder_2`, and
  builds time ids as original size, crop coordinates, and target size.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/__init__.py`
  confirms the MLX application-layer flow: tokenize prompts, run text encoders,
  repeat conditioning per image, and pass tensors into the diffusion sampler.

## Tensor Lifetime Audit

Token-id tensors are scoped to each encoder call and freed after the CLIP run.
SD 1.x / SD 2.x keeps the returned last hidden state and frees pooled outputs.
SDXL retains the selected penultimate hidden states before disposing full CLIP
outputs, frees staging hidden states after concatenation, and exposes only
caller-owned conditioning tensors. Error paths free partially assembled encoder
hidden states, text embeddings, and time ids. The returned result implements
`Disposable` and frees positive plus negative conditioning exactly once.

## Memory / Performance Evidence

This tranche adds an example composition surface and makes no throughput claim.
It does not modify CausalLM generation, transformer cache behavior, diffusion
denoising math, sampling, serving decode routes, or scheduler behavior. The
future finite image proof should measure the full text-to-image path once the
CLI can load a local Diffusers snapshot and write an image artifact.

`bench:generation` and `bench:generation:parity` were not run for this tranche:
the diff does not touch autoregressive generation hot paths.

## Independent Review

Raman reviewed the boundary and recommended keeping the first prompt-conditioning
composer in `examples/stable-diffusion/` rather than in `@mlxts/diffusion` or
`@mlxts/transformers`. The implementation follows that recommendation and keeps
package imports application-layer only.

## Validation

- `tsc -p tsconfig.phase10-examples.json`
- `bun test examples/stable-diffusion/conditioning.test.ts`
- `bun run check:phase10-proofs`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`

## Remaining Risks / Follow-ups

- The finite AXI-shaped image-generation command is still the next Stable
  Diffusion tranche. It should load the diffusion bundle and prompt conditioner,
  acquire the runtime lock, sample an image, and write an artifact.
- Real checkpoint image parity still needs a cached local Diffusers snapshot and
  reference prompt evidence before making quality claims.
