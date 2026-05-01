# LTX-2 Text Connectors Runtime Review

## Summary

This tranche adds the LTX-2 text connector runtime in `@mlxts/diffusion`. The connector accepts a Gemma hidden-state stack plus attention mask and produces separate prepared video/audio prompt embeddings for the existing LTX-2 prepared denoising path.

## Files Reviewed

- `packages/diffusion/src/families/ltx/connectors-ltx2.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-normalization.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-rotary.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-attention.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-transformer.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2.test.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/ltx2/connectors.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx2.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`

The implementation keeps the Diffusers connector split between shared/per-modality text projection, connector-local text normalization, optional learned register replacement, connector RoPE, self-attention, feed-forward residuals, and separate video/audio outputs. The connector output mask follows Diffusers' downstream binary-mask behavior: connector self-attention uses the caller mask when no learned registers are present, while the returned prepared conditioning mask is all ones after connector processing. The public runtime still accepts prepared tensors only; Gemma hidden-state extraction and checkpoint weight loading remain separate follow-up tranches.

## Tensor Lifetime Audit

Structured connector outputs retain their returned tensors and ship explicit disposal helpers. Intermediate transformer outputs are disposed after the public connector retains the video/audio prompt embeddings and attention mask it returns. Connector RoPE tensors are freed in the transformer `finally` path. Connector masks allocated before RoPE creation are released if later setup fails. Register replacement performs host-side index planning from the small attention mask, then keeps hidden/register data movement in MLX via gather and `where`.

## Memory / Performance Evidence

- `bun run --filter @mlxts/diffusion typecheck`
- `bun test packages/diffusion/src/families/ltx/connectors-ltx2.test.ts`
- `bun run lint -- packages/diffusion/src/families/ltx/connectors-ltx2.ts packages/diffusion/src/families/ltx/connectors-ltx2-transformer.ts packages/diffusion/src/families/ltx/connectors-ltx2-attention.ts packages/diffusion/src/families/ltx/connectors-ltx2-rotary.ts packages/diffusion/src/families/ltx/connectors-ltx2-normalization.ts packages/diffusion/src/families/ltx/connectors-ltx2.test.ts packages/diffusion/src/families/ltx/index.ts packages/diffusion/src/ltx.ts`

The connector runs once per prompt conditioning path, not inside the denoising step loop. No performance claim is made in this tranche.

## Independent Review

An independent explorer reviewed the diff against Diffusers for connector semantics, shape risks, tensor lifetime risks, and missing-test risks. The review identified no-register output-mask parity and pre-RoPE mask error-path ownership as issues; both were fixed before validation.

## Remaining Risks / Follow-ups

- Connector safetensor weight loading is not present yet.
- Gemma hidden-state extraction is not wired into `@mlxts/transformers` or an end-to-end LTX-2 proof command yet.
- LTX-2 transformer/VAE/audio/vocoder runtime work remains outside this tranche.
- Guidance rescale, STG, modality isolation, and prompt modulation remain outside this prepared-connector runtime.

## Out-of-scope Drift Noticed

- `PLAN.md` still under-describes the LTX-2 prepared denoising and latent-upsampling work that has already landed.
