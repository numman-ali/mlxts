# Runtime Review: LTX-2 Transformer Runtime

## Summary

This tranche adds the package-owned `Ltx2VideoTransformer3DModel` runtime for prepared packed video and audio latents plus prepared connector prompt tensors. The supported path is the current LTX-2.0 default: split or interleaved RoPE, prompt projection, dual video/audio streams, text cross-attention, cross-modality attention, and final video/audio projections.

Checkpoint loading, connector loading, Gemma hidden-state extraction, VAE decode, audio decode, vocoder integration, spatiotemporal guidance, modality isolation, prompt-modulated attention, perturbed attention, gated attention, and a proof CLI remain out of scope for this tranche.

## Files Reviewed

- `packages/diffusion/src/families/ltx/attention-ltx2.ts`
- `packages/diffusion/src/families/ltx/blocks-ltx2.ts`
- `packages/diffusion/src/families/ltx/blocks-ltx2-modulation.ts`
- `packages/diffusion/src/families/ltx/conditioning-ltx2.ts`
- `packages/diffusion/src/families/ltx/embeddings-rope.ts`
- `packages/diffusion/src/families/ltx/index.ts`
- `packages/diffusion/src/families/ltx/transformer-ltx2-final.ts`
- `packages/diffusion/src/families/ltx/transformer-ltx2-state.ts`
- `packages/diffusion/src/families/ltx/transformer-ltx2.ts`
- `packages/diffusion/src/families/ltx/transformer-ltx2.test.ts`
- `packages/diffusion/src/ltx.ts`

## Reference Audit

Reviewed local Diffusers references:

- `.reference/diffusers/src/diffusers/models/transformers/transformer_ltx2.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/connectors.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`

The runtime preserves the LTX-2 block order from the reference model: video self-attention, audio self-attention, video text cross-attention, audio text cross-attention, audio-to-video cross-attention, video-to-audio cross-attention, video feed-forward, and audio feed-forward. The cross-modal RoPE path uses temporal-only video coordinates and audio temporal coordinates, matching the reference connector between video and audio streams.

The implementation uses parameterless RMSNorm and LayerNorm paths because the current parsed LTX-2.0 configs use `norm_elementwise_affine: false`. Config branches for prompt modulation, gated attention, perturbed attention, and non-prepared prompt embeddings reject at construction until their complete runtime paths are implemented.

## Tensor Lifetime Audit

The attention path keeps projection tensors in an explicit disposable aggregate and frees projected queries, keys, values, and optional gate logits after SDPA. Block-level modulation tensors are retained only across the block and are freed after the block finishes.

The transformer path explicitly disposes RoPE tensors, timestep modulation outputs, prompt attention masks, projected prompt streams, intermediate block outputs, and final output tensors on partial-failure paths. Final video output ownership transfers only after the audio output is successfully created.

No tensor-producing primitive was hidden inside a nested expression that would obscure ownership of a native handle.

## Memory / Performance Evidence

This tranche does not claim a performance improvement. It establishes the executable model-owned baseline needed for checkpoint loading and end-to-end proof runs.

The current implementation rebuilds the four RoPE tensor sets per forward call. RoPE caching is intentionally deferred until checkpoint loading and proof benchmarks produce paired evidence for the exact reuse boundary.

## Validation

- `bun run --filter @mlxts/diffusion typecheck`
- `bun test packages/diffusion/src/families/ltx/transformer-ltx2.test.ts`
- `bun test packages/diffusion/src/families/ltx/transformer-ltx2.test.ts packages/diffusion/src/families/ltx/embeddings.test.ts packages/diffusion/src/families/ltx/pipeline-ltx2.test.ts`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run validate`

## Independent Review

Carson the 2nd reviewed the LTX-2 reference surface and recommended a narrow transformer-runtime tranche over a broader pipeline tranche. The review called out prepared packed tensors as the right boundary, the reference block ordering above, parameterless normalization for the current fixture, temporal-only video RoPE for cross-modal attention, and explicit rejection of LTX-2.3-only branches until those branches have full runtime support.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

- LTX-2 transformer safetensor weight mapping and loading.
- Connector checkpoint loading and Gemma hidden-state extraction integration.
- Video and audio autoencoder plus vocoder runtime integration.
- End-to-end LTX-2 proof CLI.
- RoPE reuse once proof benchmarks identify the correct cache boundary.
- LTX-2.3 prompt modulation, gated attention, perturbed attention, STG, and modality isolation.
