# examples/ltx-video

This example is a workbook for local LTX-Video text-to-video generation.

`@mlxts/diffusion` owns LTX scheduler, latent packing, denoising, transformer,
VAE decode, and tensor contracts.

`@mlxts/transformers` owns T5 text encoder loading.

`@mlxts/tokenizers` owns SentencePiece tokenization.

This example owns cross-package LTX prompt-conditioning composition and finite
AXI-shaped proof commands.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning and preview-frame
tests before full checkpoint video proofs.
