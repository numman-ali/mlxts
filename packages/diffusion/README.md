# `@mlxts/diffusion`

Diffusion and flow-generation primitives for mlxts.

This package is the Phase 10 counterpart to `@mlxts/transformers`: transformer
families generate autoregressively, while diffusion families iteratively denoise
latents through schedulers, backbones, VAE decoders, and conditioning tensors.

The current package surface covers the active Phase 10 image-generation ladder:
Stable Diffusion / SDXL, FLUX.1, Z-Image, Qwen-Image, FLUX.2 Klein, and
Stable Diffusion 3 / 3.5. The package owns Diffusers snapshot inspection,
scheduler/config loading, VAE and backbone modules, safetensor loading,
denoising loops, latent helpers, and conditioning tensor contracts. Prompt
tokenization, text encoders, image artifact writing, and proof-command
ergonomics stay in examples or the packages that own those encoders.

```ts
import { array } from "@mlxts/core";
import {
  EulerScheduler,
  loadDiffusionSnapshotManifest,
  loadStableDiffusionPipelineFromSnapshot,
  loadStableDiffusionComponentConfigs,
} from "@mlxts/diffusion";

const scheduler = new EulerScheduler();
const [step] = scheduler.timesteps(2);
const sample = array([0.5, -0.25], "float32");
const predictedNoise = array([0.1, -0.2], "float32");

if (step !== undefined) {
  const previous = scheduler.step(predictedNoise, sample, step);
  previous.eval();
  previous.free();
}

sample.free();
predictedNoise.free();

const manifest = await loadDiffusionSnapshotManifest("/models/stable-diffusion");
manifest.modelIndex.kind;

const configs = await loadStableDiffusionComponentConfigs(manifest);
configs.unet.crossAttentionDim;

using bundle = await loadStableDiffusionPipelineFromSnapshot("/models/stable-diffusion");
bundle.scheduler.timesteps(2);
```

## Image Support Ladder

| Family | Package Runtime | Proof Command | Real Checkpoint Evidence | Status |
| --- | --- | --- | --- | --- |
| Stable Diffusion / SDXL | VAE, UNet2D, DDIM/Euler, CFG, SD/SDXL conditioning contracts | `examples/stable-diffusion` | Official SDXL base fp16 bounded proof passed | Baseline supported path |
| FLUX.1 | FlowMatch Euler, FLUX transformer, VAE, latent packing, sampling | `examples/flux` | Official `black-forest-labs/FLUX.1-schnell` bounded proof passed | Modern flow baseline |
| Z-Image-Turbo | Dense base Z-Image transformer, FlowMatch denoising, VAE decode layout, weight loading | `examples/z-image` | Official `Tongyi-MAI/Z-Image-Turbo` bounded proof passed | Speed-first modern flow path |
| Qwen-Image / Qwen-Image-2512 | Qwen-Image transformer, 3D causal VAE, FlowMatch, true-CFG denoising, weight loading | `examples/qwen-image` | Official `Qwen/Qwen-Image-2512` bounded proof passed | Forward Qwen image-generation path |
| FLUX.2 Klein | Snapshot/config skeleton, transformer/VAE loading, prepared-embedding sampling, NCHW latent patching, external CFG, empirical FlowMatch shift, and VAE batch-norm decode boundary | `examples/flux2` | Official `black-forest-labs/FLUX.2-klein-4B` bounded proof passed | Separate family; reference-image and KV variants remain follow-ups |
| Stable Diffusion 3 / 3.5 | Snapshot/config parsing, SD3 MMDiT transformer runtime, SD3.5 dual-attention and RMS q/k norm path, transformer/VAE safetensor loading, FlowMatch denoising over prepared conditioning, VAE shift/scale decode boundary | `examples/stable-diffusion-3` | Gated official checkpoints still need authenticated proof | Runtime, loading, and prompt-conditioning foundation; finite proof command remains follow-up |

`examples/stable-diffusion`, `examples/flux`, `examples/flux2`,
`examples/z-image`, and `examples/qwen-image` own the AXI-shaped finite proof
commands that compose this package with tokenizer/text-encoder packages and
write BMP image artifacts. `examples/stable-diffusion-3` owns the SD3
application-layer CLIP/T5 prompt-conditioning bridge while its finite proof
command remains a follow-up.
SD3 authenticated checkpoint proof, FLUX.2 reference-image/KV variants,
image-to-image, inpainting, ControlNet, Omni/SigLIP, video/audio generation,
broader output formats, and quantized mflux-style sidecars remain follow-on
Phase 10 tranches until their runtime semantics are designed deliberately.
