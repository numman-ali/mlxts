# Runtime Review: FLUX Prompt Conditioning

## Summary

Added the FLUX prompt-conditioning workbook in `examples/flux` and the
tokenizer support it needs: Diffusers `spiece.model` detection, fixed-length T5
input encoding, and negative SentencePiece special-id handling. The tranche
keeps `@mlxts/diffusion` tensor-only while composing CLIP pooled projections
and T5 encoder hidden states at the example layer.

## Files Reviewed

- `packages/tokenizers/src/load.ts`
- `packages/tokenizers/src/sentencepiece-proto.ts`
- `packages/tokenizers/src/sentencepiece.ts`
- `packages/tokenizers/src/t5.ts`
- `packages/tokenizers/src/index.ts`
- `examples/flux/conditioning-runtime.ts`
- `examples/flux/conditioning-result.ts`
- `examples/flux/conditioning-types.ts`
- `examples/flux/conditioning.ts`

## Tensor Lifetime Audit

`examples/flux/conditioning-runtime.ts` frees encoded ID tensors in `finally`
blocks after CLIP and T5 encoder calls. It transfers ownership of retained
conditioning tensors to `FluxPromptConditioningResult`, which disposes
`encoderHiddenStates`, `pooledProjections`, `textIds`, and optional `guidance`
exactly once. Error paths free partially-created FLUX tensors before rethrowing.

Tokenizer changes are host-side parsing and integer-array assembly only. They do
not allocate MLX arrays or hold native handles.

## Memory / Performance Evidence

- `bun test packages/tokenizers/src/sentencepiece.test.ts packages/tokenizers/src/load.test.ts examples/flux/conditioning.test.ts`: 11 pass, 0 fail.
- `bun test packages/tokenizers/src`: 40 pass, 0 fail.
- `bun run --filter @mlxts/tokenizers typecheck`: passed.
- `bun run check:phase10-proofs`: 17 pass, 0 fail.
- `bun run check:runtime-review`: passed.
- `bun run validate`: passed.

This tranche does not modify generation loops, diffusion denoising math, model
forward hot paths, or cache behavior. Real checkpoint FLUX memory evidence is
deferred until transformer construction and weight loading exist.

## Independent Review

Turing completed a read-only architecture audit of the planned FLUX prompt
conditioning tranche. The review recommended keeping prompt composition in
`examples/flux`, keeping T5 tokenization in `@mlxts/tokenizers`, adding
`spiece.model` detection, hardening negative SentencePiece special IDs, and
leaving negative prompt / true-CFG support for a later sampler contract.

## Remaining Risks / Follow-ups

- The current SentencePiece implementation is greedy unigram matching. Before
  real FLUX checkpoint claims, add fixed Hugging Face token-ID fixtures for
  representative T5 prompts.
- FLUX transformer construction, transformer weight loading, VAE loading, and a
  finite AXI image proof command remain separate Phase 10 tranches.
- Negative prompts and true-CFG are intentionally absent until
  `@mlxts/diffusion` owns a FLUX sampler contract that consumes negative
  conditioning.
