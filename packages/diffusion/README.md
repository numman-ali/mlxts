# `@mlxts/diffusion`

Diffusion and flow-generation primitives for mlxts.

This package is the Phase 10 counterpart to `@mlxts/transformers`: transformer
families generate autoregressively, while diffusion families iteratively denoise
latents through schedulers, backbones, VAE decoders, and conditioning tensors.

The first package surface is scheduler infrastructure. It is intentionally
small and explicit so future Stable Diffusion and Flux pipelines can compose the
same scheduler contract without hiding the denoising loop.

```ts
import { array } from "@mlxts/core";
import {
  EulerScheduler,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
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

const loaded = await loadDiffusionSchedulerFromSnapshot("/models/stable-diffusion");
loaded.scheduler.timesteps(2);

const manifest = await loadDiffusionSnapshotManifest("/models/stable-diffusion");
manifest.modelIndex.kind;
```

Stable Diffusion, Flux, VAE, text conditioning, and image output examples land
as follow-on Phase 10 tranches once their reference audits and package-owned
models are in place.
