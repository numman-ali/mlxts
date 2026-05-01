# FLUX.2 Klein Transformer Runtime Review

## Files Reviewed

- `packages/diffusion/src/families/flux2/attention.ts`
- `packages/diffusion/src/families/flux2/blocks.ts`
- `packages/diffusion/src/families/flux2/embeddings.ts`
- `packages/diffusion/src/families/flux2/tensor-utils.ts`
- `packages/diffusion/src/families/flux2/transformer.ts`
- `packages/diffusion/src/families/flux2/pipeline.ts`
- `packages/diffusion/src/index.ts`

## Summary

This tranche adds a package-owned FLUX.2 Klein transformer runtime for prepared
text embeddings and packed latent sequences. It mirrors the current Diffusers
`Flux2Transformer2DModel` non-KV path: four-axis RoPE, shared timestep
modulation, double-stream image/text blocks, single-stream fused
QKV-plus-SwiGLU blocks, and final `AdaLayerNormContinuous` projection.

The tranche does not claim real checkpoint generation. Safetensor mapping,
Qwen3 prompt conditioning, reference-image KV cache, inpaint, and LoRA
processors remain separate tranches.

## Reference Check

- `.reference/diffusers` was fast-forwarded to `42a46e48c`.
- `src/diffusers/models/transformers/transformer_flux2.py` was reviewed for
  FLUX.2 block structure, modulation ownership, bias flags, RoPE axes, and
  non-KV forward order.
- `src/diffusers/pipelines/flux2/pipeline_flux2_klein.py` was reviewed for the
  prepared-embedding Klein boundary.

## Tensor Lifetime Audit

New tensor-producing operations keep local `using` scopes visible. Returned
arrays are retained explicitly where ownership crosses block boundaries. Split
projection parts and modulation chunks are freed in `finally` blocks.

`bun run check:tensor-lifetimes` passed.

## Memory / Performance Evidence

No throughput or quality claim is made. Synthetic tests cover parameter-tree
shape, four-axis ids, timestep/guidance embedding, double-stream execution,
single-stream execution, final projection shape, finite output values, and
malformed prepared-input rejection.

Commands run:

- `bun test packages/diffusion/src/families/flux2/transformer.test.ts`
- `bun test packages/diffusion/src/families/flux2`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`
- `bun run check:file-lines`
- `git diff --check`

## Independent Review

Second-pass agent review recommended a separate FLUX.2 runtime foundation,
synthetic forward tests, and explicit deferral of transformer weight loading,
Qwen3 prompt conditioning, and KV/reference-image cache support. This tranche
follows that boundary.

## Remaining Risks / Follow-ups

- Transformer safetensor loading is not wired yet.
- No real FLUX.2 Klein checkpoint proof exists until Qwen3 conditioning and
  transformer loading land.
- Reference-image KV cache behavior remains unimplemented.
- No performance claim is made; this commit establishes architecture and tensor
  correctness only.

## Out-of-scope Drift Noticed

- `.reference/transformers` is in an existing unresolved merge state and was not
  updated during this tranche.
