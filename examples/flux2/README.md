# FLUX.2 Klein Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for FLUX.2 Klein prompt conditioning and image generation.

The supported proof path loads local Diffusers `Flux2KleinPipeline` snapshots,
encodes positive and optional negative prompts through the checkpoint Qwen3 chat
template and text encoder, denoises through the FLUX.2 transformer, decodes with
`AutoencoderKLFlux2`, and writes one BMP artifact.

```bash
bun run examples/flux2/index.ts black-forest-labs/FLUX.2-klein-4B \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/flux2/flux2-klein-proof.bmp \
  --steps 4 \
  --height 256 \
  --width 256 \
  --guidance-scale 1 \
  --seed 7 \
  --dtype bfloat16 \
  --json
```

The command uses the shared runtime lock because it loads local MLX model
weights. The default tests are fixture-backed and do not require a checkpoint.
JSON output includes artifact SHA-256, BMP geometry, and non-uniform pixel
evidence. Verify a saved JSON report with:

```bash
bun run examples/image-proof/verify-report.ts .tmp/flux2/report.json \
  --expect-pipeline flux2-klein
```
