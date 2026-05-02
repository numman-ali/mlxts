# `@mlxts/diffusion`

Diffusion and flow-generation primitives for mlxts.

This package is the Phase 10 counterpart to `@mlxts/transformers`: transformer
families generate autoregressively, while diffusion families iteratively denoise
latents through schedulers, backbones, VAE decoders, and conditioning tensors.

The current package surface covers the active Phase 10 image-generation ladder
and the first video/audio snapshot entry point:
Stable Diffusion / SDXL, FLUX.1, Z-Image, Qwen-Image, FLUX.2 Klein, and
Stable Diffusion 3 / 3.5, plus LTX-Video and LTX-2 manifest/config recognition
with initial latent, RoPE geometry, classic LTX packed denoising, transformer
execution, VAE decode, classic plus LTX-2 sidecar latent upsampling, and LTX-2
prepared audio-video denoising, decode, and vocoder output.
The package owns Diffusers snapshot inspection, scheduler/config loading, VAE
and backbone modules, safetensor loading, denoising loops, latent helpers, and
conditioning tensor contracts. Prompt tokenization, text encoders, image/video
artifact writing, and proof-command ergonomics stay in examples or the packages
that own those encoders.

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
| Qwen-Image / Qwen-Image-2512 | Qwen-Image transformer, 3D causal VAE, FlowMatch, true-CFG denoising, weight loading, plus Qwen-Image Edit / Edit Plus snapshot recognition | `examples/qwen-image` | Official `Qwen/Qwen-Image-2512` bounded proof passed | Forward Qwen image-generation path; edit snapshots are recognized but edit runtime remains follow-up |
| FLUX.2 Klein | Snapshot/config skeleton, transformer/VAE loading, prepared-embedding sampling, NCHW latent patching, external CFG, empirical FlowMatch shift, and VAE batch-norm decode boundary | `examples/flux2` | Official `black-forest-labs/FLUX.2-klein-4B` bounded proof passed | Separate family; reference-image and KV variants remain follow-ups |
| Stable Diffusion 3 / 3.5 | Snapshot/config parsing, SD3 MMDiT transformer runtime, SD3.5 dual-attention and RMS q/k norm path, transformer/VAE safetensor loading, FlowMatch denoising over prepared conditioning, VAE shift/scale decode boundary | `examples/stable-diffusion-3` | Gated official checkpoints still need authenticated proof | Runtime, loading, prompt-conditioning, and finite command foundation |
| LTX-Video / LTX-2 | Diffusers `LTXPipeline`, `LTXConditionPipeline`, `LTXLatentUpsamplePipeline`, `LTX2Pipeline`, and `LTX2LatentUpsamplePipeline` manifest recognition, typed component config parsing, video latent shape/packing, LTX-2 audio latent shape/packing, video/audio RoPE coordinate helpers, classic LTX prepared-tensor packed denoising, classic LTX transformer execution/loading, classic LTX decoder-side VAE decode/loading, classic LTX latent-upsampler loading over normalized packed latents, LTX-2 latent-upsampler loading with rational spatial resampling, LTX-2 prepared audio-video denoising, video/audio VAE decode, and vocoder output | `examples/ltx-video` | Gated official checkpoint proof still needs local snapshot run | Classic LTX finite BMP-preview proof exists; LTX-2 finite proof assembly writes BMP plus PCM16 WAV evidence, with LTX-2.3 branches and real checkpoint evidence still follow-ups |

`examples/stable-diffusion`, `examples/flux`, `examples/flux2`,
`examples/z-image`, `examples/qwen-image`, and `examples/ltx-video` own the AXI-shaped finite proof
commands that compose this package with tokenizer/text-encoder packages and
write BMP image or preview artifacts. `examples/stable-diffusion-3` owns the SD3
application-layer CLIP/T5 prompt-conditioning bridge and finite proof command.
SD3 authenticated checkpoint proof, Qwen-Image edit runtime, FLUX.2
reference-image/KV variants, LTX/LTX-2 authenticated checkpoint proof,
LTX-2.3 branches, image-to-image, inpainting, ControlNet, Omni/SigLIP, broader
output formats, and quantized mflux-style sidecars remain follow-on Phase 10
tranches until their runtime semantics are designed deliberately.
