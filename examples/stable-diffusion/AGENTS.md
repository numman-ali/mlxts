# examples/stable-diffusion

This example is a workbook for local Stable Diffusion text-to-image generation.

`@mlxts/diffusion` owns VAE, UNet, scheduler, latent sampling, denoising, and
image tensor decoding.

`@mlxts/transformers` owns CLIP text encoders and weight loading.

`@mlxts/tokenizers` owns CLIP vocab/merges tokenization.

This example owns cross-package prompt-conditioning composition, local BMP
artifact I/O, and finite AXI-shaped proof commands.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast conditioning, BMP output, and command
tests. Real checkpoint proof runs use a cached local Stable Diffusion snapshot
when checkpoint files are available.
