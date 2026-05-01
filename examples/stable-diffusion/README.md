# `examples/stable-diffusion`

Local Stable Diffusion text-to-image workbook.

This example composes three package surfaces without moving the boundary into
any one package:

- `@mlxts/diffusion` loads VAE, UNet, scheduler, and samples over conditioning
  tensors.
- `@mlxts/transformers` loads CLIP text encoders.
- `@mlxts/tokenizers` loads CLIP vocab/merges tokenizers.

The prompt conditioner loads Diffusers component subfolders such as
`tokenizer/`, `text_encoder/`, `tokenizer_2/`, and `text_encoder_2/`, then
returns tensors shaped for the diffusion pipeline.

The first supported conditioning paths are:

- SD 1.x / SD 2.x: CLIP last hidden state.
- SDXL: penultimate hidden states from both CLIP encoders concatenated on the
  hidden axis, projected text embeddings from `text_encoder_2`, and six-value
  time ids for original size, crop, and target size.

Run the finite local proof command against a Diffusers-format snapshot:

```bash
bun run examples/stable-diffusion/index.ts /models/stable-diffusion \
  --prompt "a small ceramic teapot on a wooden table" \
  --output .tmp/stable-diffusion/sample.bmp
```

The command acquires the shared runtime lock, creates prompt conditioning,
samples one image through `@mlxts/diffusion`, and writes an uncompressed BMP
artifact. Progress goes to stderr; stdout is AXI-shaped structured output.
JSON output includes artifact SHA-256, BMP geometry, and non-uniform pixel
evidence. Verify a saved JSON report with:

```bash
bun run examples/image-proof/verify-report.ts .tmp/stable-diffusion/report.json \
  --expect-pipeline stable-diffusion-xl
```
