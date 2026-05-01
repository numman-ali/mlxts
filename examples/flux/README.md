# FLUX Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for FLUX prompt conditioning.

The supported proof path loads local Diffusers FLUX snapshots, encodes CLIP and
T5 prompt conditioning, denoises through the FLUX transformer, decodes with the
FLUX VAE, and writes one BMP artifact.

```bash
bun run examples/flux/index.ts /models/flux-schnell \
  --prompt "a quiet library with warm afternoon light" \
  --output .tmp/flux/sample.bmp \
  --steps 4
```

The current real-checkpoint proof target is:

```bash
bun run examples/flux/index.ts black-forest-labs/FLUX.1-schnell \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/flux/flux1-schnell-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --max-sequence-length 128 \
  --dtype float16 \
  --json
```

`FLUX.1-schnell` runs without guidance. Dev-style checkpoints with guidance
embeddings receive a default `--guidance-scale` of `3.5` unless the command
sets another non-negative value.

The command uses the shared runtime lock because it loads local MLX model
weights. The default tests are fixture-backed and do not require a checkpoint.
