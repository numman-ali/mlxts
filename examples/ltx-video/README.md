# LTX-Video Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for classic LTX-Video and LTX-2 prompt conditioning.

The supported proof paths load local Diffusers LTX snapshots, encode prompt
conditioning, denoise packed latents, decode media, and write compact proof
artifacts. Classic LTX-Video writes one BMP preview sheet. LTX-2 writes the BMP
preview plus a PCM16 WAV audio artifact.

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

The LTX-2 proof path uses the same command with an LTX-2 snapshot and writes
both preview and audio artifacts:

```bash
bun run examples/ltx-video/index.ts Lightricks/LTX-2 \
  --local-files-only \
  --prompt "a quiet library with soft ambient music" \
  --output .tmp/ltx-video/ltx2-preview.bmp \
  --audio-output .tmp/ltx-video/ltx2-audio.wav \
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
non-uniform pixel evidence. LTX-2 JSON also includes WAV SHA-256, sample rate,
duration, channel count, and sample count. Verify a saved JSON report with:

```bash
bun run examples/ltx-video/verify-report.ts .tmp/ltx-video/report.json \
  --expect-pipeline ltx-video
```
