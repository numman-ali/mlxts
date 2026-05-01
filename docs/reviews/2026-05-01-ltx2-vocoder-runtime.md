# Runtime Review: LTX-2 Vocoder Runtime

## Summary

This tranche adds the LTX-2 plain vocoder decode path in `@mlxts/diffusion`.
Decoded BCLM mel spectrogram tensors now pass through `Ltx2Vocoder` and produce
BCS waveform tensors. The implementation adds MLX-backed `convTranspose1d` in
`@mlxts/core`, exposes `ConvTranspose1d` in `@mlxts/nn`, maps Diffusers
`LTX2Vocoder` safetensors into the package module tree, and validates the
vocoder/audio-VAE sample-rate relationship.

The scope is intentionally the default LTX-2.0 plain vocoder. Bandwidth
extension, STFT/edge padding helpers, real audio artifact writing, full LTX-2
prompt/audio-video proof assembly, and LTX-2.3 branches remain separate
tranches.

## Files Reviewed

- packages/core/src/ffi/symbols.ts
- packages/core/src/index.ts
- packages/core/src/ops/index.ts
- packages/core/src/ops/linalg.ts
- packages/core/src/ops/ops.test.ts
- packages/diffusion/src/families/ltx/config-ltx2-media.ts
- packages/diffusion/src/families/ltx/config.test.ts
- packages/diffusion/src/families/ltx/config.ts
- packages/diffusion/src/families/ltx/index.ts
- packages/diffusion/src/families/ltx/vocoder-ltx2.ts
- packages/diffusion/src/families/ltx/vocoder-ltx2.test.ts
- packages/diffusion/src/families/ltx/vocoder-ltx2-weights.ts
- packages/diffusion/src/families/ltx/vocoder-ltx2-weights.test.ts
- packages/diffusion/src/ltx.ts
- packages/nn/src/index.ts
- packages/nn/src/layers/conv-transpose1d.ts
- packages/nn/src/layers/conv-transpose1d.test.ts

## Tensor Lifetime Audit

The core primitive uses the existing `readResultArrayWithMetadata` ownership
path. `ConvTranspose1d.forward()` keeps the unbiased output visible with a
`using` binding before bias addition, and frees the output if bias validation
fails.

`Ltx2Vocoder.forward()` keeps every tensor-producing step in named locals,
explicitly frees replaced hidden states, and returns only the final retained
transpose result. Resnet group outputs are collected for the parallel average
required by the reference and freed after stacking. The weight loader follows
the existing diffusion shard-iterator ownership pattern: skipped tensors are
freed immediately, transformed tensors free their source, assigned tensors move
into the module tree, and error paths free unassigned tensors.

## Memory / Performance Evidence

This change adds a required MLX primitive and model component, not an
optimization claim. The relevant performance posture is that vocoder upsampling
uses MLX `mlx_conv_transpose1d` instead of a JavaScript expansion loop.

Validation before the final gate:

```bash
bun run typecheck
bun test packages/core/src/ops/ops.test.ts packages/nn/src/layers/conv-transpose1d.test.ts packages/diffusion/src/families/ltx/vocoder-ltx2.test.ts packages/diffusion/src/families/ltx/vocoder-ltx2-weights.test.ts packages/diffusion/src/families/ltx/config.test.ts
```

The focused test run passed with 137 tests and 299 assertions before the final
review artifact pass. The full commit gate also passed:

```bash
bun run validate
```

## Independent Review

Laplace the 2nd (`019de5c4-fb6e-7750-926e-df06c9930673`) performed a read-only
second pass over the LTX-2 vocoder reference surface and recommended this
narrow tranche boundary: implement only `LTX2Vocoder`, keep BWE out of scope,
consume BCLM mel spectrograms, return BCS waveform tensors, run per-stage
resnets in parallel and average them, use PyTorch default `act_out`
LeakyReLU slope `0.01`, and translate Conv1d / ConvTranspose1d checkpoint
weights into MLX channel-last layouts.

References checked:

- `.reference/diffusers/src/diffusers/pipelines/ltx2/vocoder.py`
- `.reference/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`
- `.reference/mlx-c/mlx/c/ops.h`
- `.reference/mlx/python/mlx/nn/layers/convolution_transpose.py`
- `.reference/mlx/python/tests/test_conv_transpose.py`

## Remaining Risks / Follow-ups

- Bandwidth extension remains unsupported until STFT, edge-padding, and
  resampling helpers are implemented deliberately.
- `snake`, `snakebeta`, and antialias activation branches are rejected instead
  of approximated.
- The full LTX-2 proof still needs prompt encoding, connector invocation,
  component-bundle loading, prepared denoising orchestration, video decode,
  audio decode, vocoder decode, and real audio/video artifact writing.
- This tranche has synthetic loader/runtime coverage; real official
  checkpoint proof belongs to the full LTX-2 proof tranche.
