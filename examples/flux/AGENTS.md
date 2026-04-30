# examples/flux

This example is a workbook for local FLUX text-to-image generation.

`@mlxts/diffusion` owns FLUX scheduler, latent packing, denoising, VAE decode,
and tensor contracts.

`@mlxts/transformers` owns CLIP and T5 text encoders and weight loading.

`@mlxts/tokenizers` owns CLIP vocab/merges and SentencePiece tokenization.

This example owns cross-package FLUX prompt-conditioning composition and finite
AXI-shaped proof commands.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning tests before full
checkpoint image proofs.
