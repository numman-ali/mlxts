# Stable Diffusion VAE Weight Loading

## Summary

Added package-local Stable Diffusion VAE safetensor loading for
`@mlxts/diffusion`. The tranche maps Diffusers AutoencoderKL tensor names onto
the package-owned camelCase module tree, transforms PyTorch Conv2d kernels from
`[out, in, kh, kw]` to the channel-last MLX layout `[out, kh, kw, in]`, supports
single-shard and Diffusers index-sharded VAE weights, and fails on missing,
mismatched, or strict unexpected checkpoint tensors.

## Files Reviewed

- `packages/diffusion/src/errors.ts`
- `packages/diffusion/src/families/stable-diffusion/weights.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  confirms Diffusers VAE checkpoint key rewrites for down/up samplers, mid
  blocks, attention projections, shortcuts, and convolution kernel transposition.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/vae.py` confirms
  the MLX VAE construction that inspired the NHWC layout, while also showing
  why the example's `quant_proj`/squeeze behavior does not apply here.
- `.reference/diffusers/src/diffusers/models/autoencoders/vae.py` confirms the
  Diffusers AutoencoderKL component fields: encoder/decoder, `quant_conv`,
  `post_quant_conv`, mid block naming, GroupNorm weights, and Linear attention
  projection weights.

## Tensor Lifetime Audit

Safetensors iteration loads one tensor at a time. Skipped and unexpected tensors
are freed immediately. Transformed Conv2d tensors free their source tensor after
the contiguous channel-last copy is created. Assignment frees the existing model
parameter only after shape validation succeeds. On assignment failure, the
candidate tensor is freed before the error escapes.

## Memory / Performance Evidence

No throughput benchmark is claimed for this checkpoint-loading tranche. The
loader preserves shard-iterator-first behavior and avoids whole-checkpoint eager
materialization. Conv2d transforms use MLX transpose plus contiguous, not
host-side reshaping.

Focused validation passed:

- `bun test packages/diffusion/src/families/stable-diffusion/weights.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`

## Independent Review

Pauli completed a read-only second pass over the package VAE files and local
MLX/Diffusers references. The review confirmed the camelCase mapping, the
Conv2d-only `[0, 2, 3, 1]` transform, no Linear transposition, no 1x1 squeeze
for `quantConv` / `postQuantConv`, no up-block index reversal, and sampler
paths retaining the `.conv` segment.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

- UNet weight mapping and loading remain the next Stable Diffusion checkpoint
  tranche.
- Real checkpoint image reconstruction parity remains a later pipeline proof
  once text encoder, UNet, scheduler, and latent scaling are assembled.
