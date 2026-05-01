# examples/flux2

This example is a workbook for local FLUX.2 Klein text-to-image generation.

`@mlxts/diffusion` owns FLUX.2 scheduler, transformer, VAE, latents, denoising,
weight loading, and tensor contracts.

`@mlxts/transformers` owns Qwen3 text encoder, chat template, and weight loading.

`@mlxts/tokenizers` owns tokenizer.json tokenization.

This example owns cross-package FLUX.2 Klein prompt-conditioning composition,
local BMP artifact I/O, and finite AXI-shaped proof commands.

Reference-image, KV-cache, inpainting, and editing variants stay out of this
workbook until their package-owned runtime contracts exist.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning tests before full
checkpoint image proofs.
