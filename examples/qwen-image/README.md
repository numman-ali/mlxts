# Qwen-Image Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for Qwen-Image prompt conditioning and image generation.

The supported proof path loads local Diffusers `QwenImagePipeline` snapshots,
encodes a prompt through the checkpoint Qwen2.5-VL tokenizer and text encoder,
denoises through the Qwen-Image transformer, decodes with the 3D causal VAE,
and writes one BMP artifact.

```bash
bun run examples/qwen-image/index.ts Qwen/Qwen-Image-2512 \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --negative-prompt " " \
  --output .tmp/qwen-image/qwen-image-proof.bmp \
  --steps 4 \
  --height 1024 \
  --width 1024 \
  --true-cfg-scale 4 \
  --json
```

The current proof target is `Qwen/Qwen-Image-2512`. The command defaults to
Qwen-Image true classifier-free guidance with `--true-cfg-scale 4` and a single
space negative prompt, matching the upstream Diffusers guidance path. Increase
`--steps` for quality-oriented manual runs.

The official checkpoint has passed a bounded capability proof:

```bash
bun run examples/qwen-image/index.ts Qwen/Qwen-Image-2512 \
  --cache-dir .tmp/hf-diffusion-proof-cache \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --negative-prompt " " \
  --output .tmp/qwen-image/qwen-image-2512-official-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --true-cfg-scale 4 \
  --seed 7 \
  --dtype bfloat16 \
  --json
```

The command uses the shared runtime lock because it loads local MLX model
weights. The default tests are fixture-backed and do not require a checkpoint.
