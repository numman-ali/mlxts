# `examples/qwen3_5-image`

One-shot local image-conditioned generation for Qwen 3.5 / Qwen 3.6 multimodal
checkpoints.

This example keeps file decode and resize outside `@mlxts/transformers` on
purpose. The package owns checkpoint loading, multimodal prompt preparation,
and tensor-level image patchification; the example owns local image I/O.

On Apple Silicon, the most practical first smoke target is the quantized MLX
conversion:

```bash
bun run examples/qwen3_5-image/index.ts mlx-community/Qwen3.6-27B-4bit \
  --image ./photo.jpg \
  --prompt "Describe what is happening in this image." \
  --greedy
```

You can also point it at a local snapshot directory or another compatible Qwen
3.5 multimodal conversion. Hub sources use cached files by default; pass
`--allow-download` when the snapshot is not already present locally. The example
will:

- resolve the snapshot through the official Hugging Face JS client
- load the model, tokenizer, chat template, and `preprocessor_config.json`
- resize the image with the checkpoint's smart-resize policy
- decode and resize the local image with macOS `sips`
- patchify the decoded RGB image into `pixel_values` and `image_grid_thw`
- prepare a Qwen multimodal prompt and run one-shot generation

Arguments:

```bash
bun run examples/qwen3_5-image/index.ts <model-path-or-repo-id> --image <path> \
  [--prompt <text>] [--system-prompt <text>] [--max-tokens <n>] \
  [--temperature <n>] [--top-k <n>] [--top-p <n>] [--greedy] \
  [--enable-thinking|--disable-thinking|--template-default-thinking] \
  [--allow-download] [--json]
```

Thinking is disabled by default for short visual descriptions. Use
`--enable-thinking` or `--template-default-thinking` when testing model-native
reasoning behavior.

The default finite command output is compact structured stdout. Model loading,
image resize status, and other progress lines go to stderr. Usage errors emit
structured stdout and exit `2`; runtime failures emit structured stdout and exit
`1`. `--json` preserves JSON result stdout compatibility while keeping progress
on stderr.

`sips` is required because this repo targets Apple Silicon and keeps local
image decode out of the package core.
