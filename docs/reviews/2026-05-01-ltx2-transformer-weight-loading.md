# LTX-2 Transformer Weight Loading Review

## Summary

LTX-2 transformer safetensor loading now maps the Diffusers checkpoint tree onto the package-owned prepared transformer runtime. The tranche is a structural loading slice only; it does not change denoising math or enable unsupported LTX-2.3 branches.

## Files Reviewed

- `packages/diffusion/src/families/ltx/transformer-ltx2-weights.ts`
- `packages/diffusion/src/families/ltx/transformer-ltx2-weights.test.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`

## Reference Audit

- Diffusers `LTX2VideoTransformer3DModel` names video/audio input projections, prompt projections, six `LTX2AdaLayerNormSingle` modules, paired video/audio block attention modules, audio-to-video and video-to-audio attention modules, paired feed-forwards, per-block cross-modality modulation tables, and final projections.
- The TypeScript runtime supports the LTX-2.0 prepared tensor path only. The weight mapper leaves LTX-2.3 prompt modulation, gated attention, affine Q/K norms, affine output norms, dropout modules, and perturbed-attention-only tensors unmapped so strict loading reports them as unexpected.
- Linear and table tensors use the same shape convention as local `Linear` and direct `MxArray` parameters. No checkpoint tensor transform is required for the supported branch.

## Tensor Lifetime Audit

- Safetensor tensors are either assigned into the model tree or freed immediately when unmapped.
- Assignment frees the previous model parameter only after shape validation succeeds.
- Transformed tensors are tracked through a nullable local and freed on assignment failure.
- Snapshot construction disposes the partially loaded model on load failure.

## Validation

- `bun test packages/diffusion/src/families/ltx/transformer-ltx2-weights.test.ts`
- `bun run typecheck`
- `bun run check:file-lines`

## Memory / Performance Evidence

- No generation or denoising hot path changed.
- Focused loader tests covered full-shard load, indexed-shard load, missing weight rejection, shape mismatch rejection, strict unexpected tensor rejection, and partial snapshot disposal.
- The implementation keeps tensor assignment one tensor at a time through `iterateSafetensors`; it does not materialize whole shards or add additional parameter copies.

## Independent Review

- Carson the 2nd audited the Diffusers LTX-2 checkpoint names and confirmed the separate LTX-2 mapper, no-transform tensor path, unsupported-branch rejection, and synthetic loader coverage plan.

## Remaining Risks / Follow-ups

- Real checkpoint loading still depends on the rest of the LTX-2 pipeline weight loaders for connector, VAE, audio VAE, and vocoder components. This tranche only makes the transformer component loadable.
