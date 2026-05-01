# examples/z-image

This example is a workbook for local Z-Image text-to-image generation.

`@mlxts/diffusion` owns Z-Image scheduler, transformer, VAE, latents,
denoising, and tensor contracts.

`@mlxts/transformers` owns Qwen3 text encoder, chat template, and weight
loading.

`@mlxts/tokenizers` owns tokenizer.json tokenization.

This example owns cross-package Z-Image prompt-conditioning composition, local
BMP artifact I/O, and finite AXI-shaped proof commands.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning tests before full
checkpoint image proofs.
