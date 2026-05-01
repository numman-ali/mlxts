# Runtime Review: LTX-2 Text Connector Weight Loading

## Summary

This tranche adds Diffusers safetensor loading for the LTX-2 `LTX2TextConnectors`
component. The loader maps shared or per-modality text projections, video/audio
connector learnable registers, connector attention q/k RMSNorm weights,
attention projections, optional gate logits, and connector feed-forward weights
onto the package-owned `Ltx2TextConnectors` module tree.

The tranche does not add LTX-2 prompt encoding, pipeline assembly, audio/video
autoencoder execution, vocoder execution, or LTX-2.3-specific behavior beyond
loading parameters that the existing connector runtime already owns.

## Files Reviewed

- `packages/diffusion/src/families/ltx/connectors-ltx2.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-attention.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-transformer.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2.test.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-weights.ts`
- `packages/diffusion/src/families/ltx/connectors-ltx2-weights.test.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/ltx.ts`

Reference files:

- `.reference/diffusers/src/diffusers/pipelines/ltx2/connectors.py`
- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx2.py`

## Tensor Lifetime Audit

Safetensor iteration transfers one tensor at a time. Unmapped tensors are freed
immediately. Assigned tensors replace existing module parameters only after a
shape check, and the old parameter is freed at assignment time. The transform
hook is identity for current connector tensors; if a future connector tensor
requires a layout transform, ownership remains isolated inside
`assignWeightTensor`.

Snapshot construction disposes partially loaded connector modules on failure.
Focused tests cover missing weights, shape mismatch, strict unexpected weights,
plain safetensor shards, indexed shards, snapshot-manifest loading, and the
per-modality projection hidden-dimension guard.

## Memory / Performance Evidence

No generation hot path changes in this tranche. Loading remains shard-iterator
based and does not materialize whole safetensor shards. Connector runtime tensor
execution is unchanged.

Focused validation commands:

```bash
bun test packages/diffusion/src/families/ltx/connectors-ltx2-weights.test.ts
```

## Independent Review

Chandrasekhar the 2nd performed a read-only second-pass audit of the LTX-2
connector reference and local module tree. The review confirmed the intended
mapping surface: top-level text projections, `learnable_registers`,
`transformer_blocks.*.attn1` q/k RMSNorm plus q/k/v/out/gate projections, and
`ff.net.{0.proj,2}` feed-forward tensors.

## Remaining Risks / Follow-ups

- LTX-2 video/audio autoencoder and vocoder loading/execution remain separate
  product tranches before a real audio-video proof can run end to end.
- LTX-2.3 prompt modulation and advanced connector variants remain bounded by
  the existing runtime support, not by checkpoint-name recognition alone.
