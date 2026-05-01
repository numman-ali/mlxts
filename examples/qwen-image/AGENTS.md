# examples/qwen-image

This example is a workbook for local Qwen-Image text-to-image generation.

`@mlxts/diffusion` owns Qwen-Image scheduler, transformer, VAE, latents,
denoising, and tensor contracts.

`@mlxts/transformers` owns Qwen2.5-VL text encoder, tokenizer loading, and
checkpoint weight loading.

This example owns cross-package Qwen-Image prompt-conditioning composition,
local BMP artifact I/O, and finite AXI-shaped proof commands.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning tests before full
checkpoint image proofs.
