# `@mlxts/diffusion`

Diffusion and flow-generation primitives for mlxts.

This package is the Phase 10 counterpart to `@mlxts/transformers`: transformer
families generate autoregressively, while diffusion families iteratively denoise
latents through schedulers, backbones, VAE decoders, and conditioning tensors.

The current package surface covers scheduler infrastructure plus the Stable
Diffusion / SDXL package path: local Diffusers manifest/config inspection,
VAE/UNet construction and safetensor loading, scheduler-backed sampling, and a
disposable pipeline bundle that accepts caller-owned conditioning tensors.
Prompt tokenization, CLIP text encoding, and image artifact writing stay outside
this package boundary.

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

`examples/stable-diffusion` owns the AXI-shaped finite proof command that
composes this package with CLIP tokenizers/text encoders and writes a BMP image
artifact. Real checkpoint image evidence, FLUX.1, Qwen-Image, Z-Image-Turbo,
Stable Diffusion 3 / 3.5, Hub-backed snapshot resolution, image-to-image,
inpainting, and broader output formats remain follow-on Phase 10 tranches.
