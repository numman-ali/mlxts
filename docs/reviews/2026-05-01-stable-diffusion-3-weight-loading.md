# Runtime Review: Stable Diffusion 3 Weight Loading

## Summary

Implemented Stable Diffusion 3 / 3.5 transformer and VAE safetensor loading for inspected Diffusers snapshots. The tranche adds package-owned checkpoint name mapping, tensor layout transforms, strict unexpected-weight handling, thin SD3 VAE shift metadata, and generated local snapshot proofs for both base SD3 and SD3.5-style transformer shapes.

No real image-generation or throughput claim is made. Official Stability checkpoints remain gated, so this tranche proves assignment and snapshot mechanics with generated safetensors only.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion-3/autoencoder.ts`
- `packages/diffusion/src/families/stable-diffusion-3/weight-mapping.ts`
- `packages/diffusion/src/families/stable-diffusion-3/weights.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- Local Diffusers was refreshed before implementation; `.reference/diffusers` fast-forwarded to `ffd5da5f7`.
- `.reference/diffusers/src/diffusers/models/transformers/transformer_sd3.py` confirms `SD3Transformer2DModel`, patch projection naming, final projection naming, and the fixed `pos_embed.pos_embed` buffer.
- `.reference/diffusers/src/diffusers/models/attention.py` and `.reference/diffusers/src/diffusers/models/attention_processor.py` confirm SD3 joint attention naming, context projections, `to_add_out`, optional `attn2`, and q/k norm tensor names.
- `.reference/diffusers/src/diffusers/models/normalization.py` confirms AdaLN-Zero, AdaLN-Continuous, and SD3.5 AdaLN-Zero-X modulation module naming.
- `.reference/diffusers/src/diffusers/models/embeddings.py` confirms combined timestep/text embedder naming.
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py` confirms the reused AutoencoderKL VAE component shape and quant/post-quant Conv2d projections.

## Tensor Lifetime Audit

Safetensors are still iterated one tensor at a time. Assigned tensors transfer ownership into the module tree after shape validation. Transformed Conv2d tensors free the source checkpoint tensor after creating the contiguous transposed tensor. Ignored fixed SD3 positional buffers are freed immediately. Unexpected tensors are freed before strict handling. Snapshot constructors dispose partially loaded models on failure.

## Coverage

Generated safetensor tests cover:

- SD3 VAE mapping and Conv2d layout transforms, including quant and post-quant projections.
- SD3 VAE snapshot loading and `shiftFactor` metadata preservation.
- SD3.5 transformer loading with RMS q/k norms, dual attention, final context-pre-only shape, and strict skipped `pos_embed.pos_embed`.
- Base SD3 transformer loading with no q/k norms and no dual attention.
- Missing, mismatched, and strict unexpected checkpoint tensors for both VAE and transformer paths.

## Memory / Performance Evidence

No generation benchmark was run because this tranche changes loading and assignment, not denoising execution. The loader preserves the existing shard-iterator memory posture and avoids whole-shard eager materialization.

## Independent Review

Boole the 2nd (`019de3c5-27c1-7e42-8b6a-044ed0bc04a3`) performed a read-only second pass over the SD3 loading tranche. The review confirmed the correct mapping boundary, called out `pos_embed.pos_embed` as a required known fixed buffer skip, required both SD3 base and SD3.5 q/k norm plus dual-attention coverage, and recommended deferring prompt encoders and real checkpoint proof.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Official SD3 / SD3.5 checkpoint proof still requires authenticated access to gated Stability snapshots. Prompt conditioning still belongs in the application layer over two CLIP projection encoders, one T5 encoder, and three tokenizers. Real image quality, throughput, LoRA, ControlNet, skip-layer guidance, img2img, inpainting, and PAG variants remain separate tranches.
