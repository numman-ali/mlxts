# Stable Diffusion UNet Weight Loading

## Summary

Added package-local Stable Diffusion UNet safetensor loading for
`@mlxts/diffusion`. The tranche maps Diffusers `UNet2DConditionModel` tensor
names onto the package-owned camelCase module tree, transforms PyTorch Conv2d
kernels from `[out, in, kh, kw]` to `[out, kh, kw, in]`, keeps Linear and Norm
weights unchanged, supports single-shard and Diffusers index-sharded UNet
weights, and fails on missing, mismatched, or strict unexpected checkpoint
tensors.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/config.ts`
- `packages/diffusion/src/families/stable-diffusion/weights.ts`
- `packages/diffusion/src/index.ts`

## Reference Audit

- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  confirms Stable Diffusion UNet checkpoint key rewrites for down/up samplers,
  mid blocks, attention projections, transformer projections, and Conv2d kernel
  transposition.
- `.reference/diffusers/src/diffusers/models/unets/unet_2d_condition.py`
  confirms `time_embedding`, optional `add_embedding`, down/mid/up block
  topology, `use_linear_projection`, and the non-null
  `time_embedding_act_fn` / `timestep_post_act` behavior that now rejects during
  config parsing.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py`
  confirms Diffusers public UNet checkpoint names for time embeddings, down
  blocks, mid block, up blocks, and sampler convolution paths.

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
- `bun test packages/diffusion/src/families/stable-diffusion/config.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run validate`

## Independent Review

Hubble completed a read-only second pass over current `@mlxts/diffusion` files
and local MLX/Diffusers references. The review confirmed that Diffusers
`up_blocks` indices must remain in checkpoint order, Conv2d weights transpose
without squeezing 1x1 projection kernels, Linear/Norm tensors stay unchanged,
and `ff.net.0.proj` must load into the fused `feedForward.projectionIn` rather
than being split.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

- Full Stable Diffusion pipeline assembly remains the next checkpoint tranche:
  text encoder loading, tokenizer conditioning, scheduler loop, VAE latent
  scaling, and image decode parity are still separate product work.
- Real checkpoint image parity is not claimed until the full text-to-image
  pipeline exists and can compare generated output against a reference prompt.
