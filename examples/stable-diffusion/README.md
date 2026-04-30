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

The finite image-generation command lands after this conditioning surface, so
real checkpoint proof can load one local Diffusers snapshot, create conditioning,
sample an image, and write an artifact.
