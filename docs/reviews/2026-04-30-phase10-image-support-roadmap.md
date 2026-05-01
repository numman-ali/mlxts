# Phase 10 Image Support Roadmap

## References Reviewed

- `PLAN.md`
- `docs/gates-and-milestones.md`
- `docs/python-equivalence-map.md`
- `continuity.md`
- `MEMORY.md`
- `.reference/diffusers/docs/source/en/api/pipelines/flux.md`
- `.reference/diffusers/docs/source/en/api/pipelines/qwenimage.md`
- `.reference/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`
- `.reference/diffusers/docs/source/en/api/pipelines/z_image.md`
- `.reference/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image.py`
- `.reference/diffusers/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md`
- `.reference/diffusers/src/diffusers/pipelines/auto_pipeline.py`

## Current Support State

Stable Diffusion / SDXL is the first image-generation baseline in
`@mlxts/diffusion`: local Diffusers snapshot inspection, VAE/UNet
construction and loading, scheduler loading, sampling, pipeline loading, and an
AXI-shaped proof command exist. Real checkpoint image evidence remains pending.

FLUX.1 is now represented by package-owned FlowMatch Euler scheduling, FLUX
transformer config/backbone/weight loading, FLUX VAE config/loading/decoding,
latent packing, sampling, and an AXI-shaped `examples/flux` proof command. The
first target stays `black-forest-labs/FLUX.1-schnell`; its timestep-distilled
constraints stay explicit at the example boundary.

Qwen image understanding is already represented through
`@mlxts/transformers`, `@mlxts/serve`, and `examples/qwen3_5-image`. Qwen-Image
generation is a separate diffusion/flow family and belongs in
`@mlxts/diffusion`.

## Support Order

1. Stable Diffusion / SDXL real checkpoint proof.
2. FLUX.1 real checkpoint proof and operator docs for gated/non-commercial
   variants.
3. Qwen-Image reference audit and config/model-index parsing. The family uses
   FlowMatch Euler, `QwenImageTransformer2DModel`, `AutoencoderKLQwenImage`,
   and Qwen2.5-VL text encoding. The Qwen image VAE is 3D causal and
   Qwen/Wan-derived, so it must not be implemented as a thin Stable Diffusion
   or FLUX AutoencoderKL reuse.
4. Z-Image-Turbo reference audit and finite proof command. Diffusers exposes it
   as a 6B `ZImageTransformer2DModel` pipeline over FlowMatch Euler, standard
   `AutoencoderKL`, and chat-template prompt encoding; the Turbo checkpoint's
   eight-step target makes it the first speed-first local image loop after
   Qwen-Image is structurally represented.
5. Stable Diffusion 3 / 3.5 and distilled variants after the shared MMDiT/flow
   seams can support `SD3Transformer2DModel`, FlowMatch Euler, AutoencoderKL,
   and three text encoders including T5-XXL without creating a parallel package
   shape.

## Decision

The roadmap keeps support ordered by architectural leverage, not launch
excitement. Stable Diffusion / SDXL and FLUX prove the package surface and flow
sampling path. Qwen-Image comes before Z-Image because it has a distinct VAE
and Qwen2.5-VL text-encoder requirement that would be expensive to retrofit
later. Z-Image-Turbo follows because it is the best local product-speed target
once the Qwen/flow-DiT seams are honest. SD3/3.5 follows once the shared MMDiT
surface prevents a one-family fork.

## Next Implementation Gate

The next code tranche is Qwen-Image reference skeleton work: model-index
classification, component config parsing, and focused tests that prove the
family is recognized without pretending the transformer, VAE, or text encoder
is already runnable. Runtime tensor code waits for the follow-up review
artifact that names the Qwen-Image transformer, VAE, scheduler, text encoder,
and tokenizer files under review.

## Out-of-scope Drift Noticed

Diffusers main already lists additional image families such as Flux2, Ovis,
NucleusMoE Image, and Hunyuan image/video. They are intentionally outside the
current Phase 10 ladder until the core SD/FLUX/Qwen/Z/SD3 surfaces have real
checkpoint evidence and shared infrastructure.
