# LTX-Video Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for classic LTX-Video prompt conditioning.

The supported proof path loads local Diffusers LTX-Video snapshots, encodes T5
prompt conditioning, denoises packed video latents through the LTX transformer,
decodes with the LTX VAE, and writes one BMP preview-frame artifact.

```bash
bun run examples/ltx-video/index.ts /models/ltx-video \
  --prompt "a quiet library with warm afternoon light" \
  --output .tmp/ltx-video/preview.bmp \
  --steps 4
```

The current finite checkpoint proof target is:

```bash
bun run examples/ltx-video/index.ts Lightricks/LTX-Video \
  --local-files-only \
  --prompt "a small red apple on a white table, cinematic video" \
  --output .tmp/ltx-video/ltx-video-proof.bmp \
  --steps 2 \
  --height 128 \
  --width 128 \
  --frames 9 \
  --max-sequence-length 128 \
  --dtype float16 \
  --json
```

The command uses the shared runtime lock because it loads local MLX model
weights. The default tests are fixture-backed and do not require a checkpoint.
JSON output includes preview-frame SHA-256, BMP geometry, video geometry, and
non-uniform pixel evidence. Verify a saved JSON report with:

```bash
bun run examples/image-proof/verify-report.ts .tmp/ltx-video/report.json \
  --expect-pipeline ltx-video
```
