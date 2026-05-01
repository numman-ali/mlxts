# FLUX.2 Klein Autoencoder Loading

## Summary

Added package-owned `AutoencoderKLFlux2` construction and safetensor loading
for FLUX.2 Klein. The tranche preserves Diffusers' NCHW VAE boundary at the
FLUX.2 family surface, reuses the existing channel-last VAE internals through
explicit transposes, supports `decoder_block_out_channels` for small-decoder
snapshots, and loads `bn.running_mean` / `bn.running_var` as required
non-parameter checkpoint buffers.

This is not a full FLUX.2 image proof. It intentionally does not add
`Flux2Transformer2DModel` execution, transformer weight loading, Qwen3 prompt
conditioning, KV/reference variants, or an example command.

References checked:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/5e67da950fce4a097bc150c22958a05716994cea/vae/config.json
- https://huggingface.co/black-forest-labs/FLUX.2-small-decoder
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`
- `.reference/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`
- `.reference/diffusers/src/diffusers/models/autoencoders/vae.py`

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/config.ts`
- `packages/diffusion/src/families/stable-diffusion/autoencoder.ts`
- `packages/diffusion/src/families/flux2/autoencoder.ts`
- `packages/diffusion/src/families/flux2/weights.ts`
- `packages/diffusion/src/index.ts`

## Tensor Lifetime Audit

The FLUX.2 wrapper keeps all layout conversions as named `using` intermediates.
`encodeMoments`, `decode`, and `forward` return newly owned tensors and dispose
the temporary channel-layout tensors before returning. The loader either assigns
a transformed tensor into the module tree or frees it on failure; batch-norm
buffer tensors are copied to host arrays and freed in `finally`.

## Memory / Performance Evidence

No new denoising transformer hot path or real checkpoint proof was added, so no
throughput or quality claim is made. Synthetic tests cover NCHW encode/decode
shape boundaries, channel-axis posterior splitting, small-decoder channel
topology, copied batch-norm statistics, complete single-shard loading,
safetensors index loading, strict unexpected-weight rejection, missing required
buffer rejection, and batch-norm shape mismatch rejection.

Commands run:

- `bun test packages/diffusion/src/families/flux2`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`

## Independent Review

Russell the 2nd performed a read-only GPT-5.5 xhigh review of the local
implementation and Diffusers references. The review recommended this tranche
before transformer runtime work, specifically because the existing FLUX.2 decode
contract already depends on `latentChannels`, batch-norm statistics, and NCHW
VAE decode semantics.

## Remaining Risks / Follow-ups

- `Flux2Transformer2DModel` runtime execution and transformer safetensor weight
  mapping remain future work.
- Qwen3 prompt conditioning remains future work, including hidden-state
  selection, tokenizer/chat-template behavior, and example-owned proof
  ergonomics.
- Reference-image, inpainting, KV-cached, GGUF/single-file, LoRA, quantized,
  and small-decoder standalone snapshot composition remain unsupported.
- No `examples/flux2` proof command or real checkpoint image artifact exists
  yet.

## Out-of-scope Drift Noticed

- BFL now publishes a separate FLUX.2 small decoder with narrower
  `decoder_block_out_channels`. This tranche supports the topology, but it does
  not add a user-facing composition path for substituting that decoder into a
  Klein snapshot.
