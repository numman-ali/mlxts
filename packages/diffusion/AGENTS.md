# @mlxts/diffusion

`@mlxts/diffusion` owns diffusion and flow generation: schedulers, denoising
backbones, VAE modules, conditioning contracts, sampling loops, diffusion
checkpoint loading, and diffusion-specific fine-tuning surfaces.

Autoregressive media understanding stays in `@mlxts/transformers`. Conditioning
tensors enter this package as tensors or package-local conditioning values;
`@mlxts/diffusion` does not import `@mlxts/transformers`.

Schedulers expose explicit timesteps, sigma or alpha schedules, and tensor step
functions. Hidden global denoising state is forbidden. Sampling loops pass the
current timestep and model output through visible control flow.

Backbone and VAE modules extend `Module`. Config values, scheduler state, and
non-parameter runtime caches stay outside enumerable module fields.

Stable Diffusion, Flux, video, and audio families use intent-named folders.
Shared scheduler and sampling infrastructure lives under `src/schedulers/` and
`src/sampling/`; family-specific checkpoint translation stays under
`src/families/<name>/`.

Runtime-sensitive tensor code requires a paired review artifact under
`docs/reviews/` and focused tests before full validation.
