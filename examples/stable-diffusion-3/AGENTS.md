# examples/stable-diffusion-3

This example is a workbook for local Stable Diffusion 3 / 3.5 text-to-image generation.

`@mlxts/diffusion` owns SD3 transformer, VAE, scheduler, latent sampling,
denoising, and prepared conditioning tensor contracts.

`@mlxts/transformers` owns CLIP projection encoders, T5 encoders, and weight
loading.

`@mlxts/tokenizers` owns CLIP vocab/merges and T5 SentencePiece tokenization.

This example owns cross-package SD3 prompt-conditioning composition, local image
artifact I/O, and finite AXI-shaped proof commands.

IP-Adapter, ControlNet, skip-layer guidance, img2img, inpainting, and PAG
variants stay out of this workbook until their package-owned runtime contracts
exist.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast prompt-conditioning tests before full
checkpoint image proofs.
