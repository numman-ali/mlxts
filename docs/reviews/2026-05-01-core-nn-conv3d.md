# Runtime Review: Core and NN Conv3d

## Summary

Added MLX-backed 3D convolution to the core tensor surface and `@mlxts/nn` layer surface. The tranche binds the existing `mlx_conv3d` primitive through the package native adapter, exposes channel-last `conv3d`, adds `Conv3d`, and keeps scope to `groups=1` because the vendored MLX convolution backend rejects grouped 3D convolution.

## Files Reviewed

- `packages/core/native/mlxts_core_ops.cpp`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/ops.test.ts`
- `packages/nn/src/layers/conv3d.ts`
- `packages/nn/src/layers/conv3d.test.ts`
- `packages/nn/src/index.ts`
- `packages/nn/AGENTS.md`
- `packages/nn/README.md`
- `scripts/runtime-sensitive-ops.ts`

## ABI / Reference Audit

Vendored `mlx-c` exposes `mlx_conv3d(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, stream)`. The native adapter passes a packed parameter array in that order. MLX reference tests and docs use channel-last input `[N, D, H, W, C]` and weight `[out, kd, kh, kw, in / groups]`.

Vendored MLX rejects `groups != 1` for rank-5 convolution, so both `conv3d` and `Conv3d` reject grouped 3D convolution explicitly instead of surfacing a misleading public capability.

## Tensor Lifetime Audit

`conv3d` follows the existing `conv2d` result-reader pattern and returns a single owned `MxArray`. `Conv3d.forward` keeps the convolution output visible before optional bias addition, disposes it through `using` when bias is applied, and frees the output on bias-shape failure. `conv3d` is included in `TRACKED_TENSOR_PRODUCING_CALL_NAMES`.

## Memory / Performance Evidence

The native dylib was rebuilt after adding `mlxts_conv3d`. Focused tests passed:

```bash
bun test packages/core/src/ops/ops.test.ts packages/nn/src/layers/conv3d.test.ts
```

The focused tests cover value correctness, stride/padding triples, grouped-conv rejection, layer parameter scanning, bias application, replacement-shape validation, and gradient flow through `Conv3d`.

`bun run typecheck`, `bun run lint`, `bun run check:tensor-lifetimes`, `bun run check:runtime-review`, and `bun run check:coverage` passed before commit.

No performance claim is made. This tranche exposes the MLX primitive needed by Qwen-Image and other 3D VAE paths without adding a custom kernel or JS fallback.

## Independent Review

Ohm (`019de1cd-47e8-7553-aa74-1973f28df0ad`) performed a read-only Conv3d audit against vendored `mlx-c`, `.reference/mlx`, and existing `Conv1d` / `Conv2d` patterns. The review confirmed the ABI order, channel-last layout, weight layout, required files, tensor-lifetime tracking requirement, and the grouped-Conv3d limitation.

## Remaining Risks / Follow-ups

Diffusers and PyTorch Conv3d weights are channel-first `[out, in, kd, kh, kw]`; Qwen-Image VAE loading must transpose them to MLX layout before assignment. Generic `Conv3d` does not implement causal temporal padding, temporal cache stitching, or Qwen/Wan-specific video/image VAE policy; those belong in the family layer composition.

## Out-of-scope Drift Noticed

None.
