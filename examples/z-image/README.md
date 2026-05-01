# Z-Image Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for Z-Image prompt conditioning and image generation.

The supported proof path loads local Diffusers `ZImagePipeline` snapshots,
encodes a prompt through the checkpoint Qwen3 chat template and text encoder,
denoises through the Z-Image transformer, decodes with the VAE, and writes one
BMP artifact.

```bash
bun run examples/z-image/index.ts Tongyi-MAI/Z-Image-Turbo \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/z-image/z-image-turbo-proof.bmp \
  --steps 9 \
  --height 1024 \
  --width 1024 \
  --guidance-scale 0 \
  --json
```

The current proof target is `Tongyi-MAI/Z-Image-Turbo`. Turbo generation runs
with `--guidance-scale 0` because this example does not implement classifier-free
guidance yet.

The command uses the shared runtime lock because it loads local MLX model
weights. The default tests are fixture-backed and do not require a checkpoint.
