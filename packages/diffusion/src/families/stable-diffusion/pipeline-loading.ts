/**
 * Stable Diffusion snapshot-to-runtime bundle loading.
 * @module
 */

import type { MxArray } from "@mlxts/core";

import { DiffusionConfigError } from "../../errors";
import {
  createDiffusionScheduler,
  type ParsedDiffusionSchedulerConfig,
} from "../../pretrained/scheduler-config";
import {
  type DiffusionSnapshotManifest,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import type { StableDiffusionAutoencoderKL } from "./autoencoder";
import {
  loadStableDiffusionComponentConfigs,
  type StableDiffusionComponentConfigs,
} from "./config";
import {
  decodeStableDiffusionLatents,
  denoiseStableDiffusionLatents,
  generateStableDiffusionImage,
  type StableDiffusionDenoiseOptions,
  type StableDiffusionImageGenerationOptions,
  type StableDiffusionScheduler,
} from "./pipeline";
import type { StableDiffusionUNet2DConditionModel } from "./unet";
import {
  loadStableDiffusionAutoencoderFromSnapshot,
  loadStableDiffusionUNetFromSnapshot,
  type StableDiffusionAutoencoderWeightLoadOptions,
  type StableDiffusionUNetWeightLoadOptions,
} from "./weights";

/** Weight-loading controls for Stable Diffusion runtime bundle construction. */
export type StableDiffusionPipelineLoadOptions = {
  autoencoder?: StableDiffusionAutoencoderWeightLoadOptions;
  unet?: StableDiffusionUNetWeightLoadOptions;
};

/** Loaded Stable Diffusion runtime components for sampling over supplied conditioning tensors. */
export type StableDiffusionPipelineBundle = Disposable & {
  readonly snapshotDirectory: string;
  readonly manifest: DiffusionSnapshotManifest;
  readonly configs: StableDiffusionComponentConfigs;
  readonly vae: StableDiffusionAutoencoderKL;
  readonly unet: StableDiffusionUNet2DConditionModel;
  readonly scheduler: StableDiffusionScheduler;
  generateImage(
    options: Omit<StableDiffusionImageGenerationOptions, "unet" | "vae" | "scheduler">,
  ): MxArray;
  denoiseLatents(options: Omit<StableDiffusionDenoiseOptions, "unet" | "scheduler">): MxArray;
  decodeLatents(latents: MxArray): MxArray;
};

class LoadedStableDiffusionPipelineBundle implements StableDiffusionPipelineBundle {
  #disposed = false;

  constructor(
    readonly snapshotDirectory: string,
    readonly manifest: DiffusionSnapshotManifest,
    readonly configs: StableDiffusionComponentConfigs,
    readonly vae: StableDiffusionAutoencoderKL,
    readonly unet: StableDiffusionUNet2DConditionModel,
    readonly scheduler: StableDiffusionScheduler,
  ) {}

  generateImage(
    options: Omit<StableDiffusionImageGenerationOptions, "unet" | "vae" | "scheduler">,
  ): MxArray {
    this.assertOpen();
    return generateStableDiffusionImage({
      ...options,
      unet: this.unet,
      vae: this.vae,
      scheduler: this.scheduler,
    });
  }

  denoiseLatents(options: Omit<StableDiffusionDenoiseOptions, "unet" | "scheduler">): MxArray {
    this.assertOpen();
    return denoiseStableDiffusionLatents({
      ...options,
      unet: this.unet,
      scheduler: this.scheduler,
    });
  }

  decodeLatents(latents: MxArray): MxArray {
    this.assertOpen();
    return decodeStableDiffusionLatents(this.vae, latents);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.unet[Symbol.dispose]();
    this.vae[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("StableDiffusionPipelineBundle has been disposed.");
    }
  }
}

function assertUnsupportedSafetyCheckerAbsent(manifest: DiffusionSnapshotManifest): void {
  const requiresSafetyChecker = manifest.modelIndex.pipelineConfig.requires_safety_checker;
  if (requiresSafetyChecker !== undefined && requiresSafetyChecker !== false) {
    throw new DiffusionConfigError(
      "Stable Diffusion pipeline loading requires requires_safety_checker=false.",
    );
  }
  const safetyChecker = manifest.components.find(
    (component) => component.name === "safety_checker",
  );
  if (safetyChecker?.enabled === true) {
    throw new DiffusionConfigError(
      "Stable Diffusion pipeline loading does not support enabled safety_checker components yet.",
    );
  }
}

function createStableDiffusionScheduler(
  parsedConfig: ParsedDiffusionSchedulerConfig,
): StableDiffusionScheduler {
  if (parsedConfig.kind === "ddim") {
    return createDiffusionScheduler(parsedConfig);
  }
  if (parsedConfig.kind === "euler") {
    return createDiffusionScheduler(parsedConfig);
  }
  throw new DiffusionConfigError(
    "Stable Diffusion pipeline loading requires a DDIM or Euler scheduler.",
  );
}

/** Load Stable Diffusion VAE, UNet, scheduler, and parsed metadata from a local snapshot. */
export async function loadStableDiffusionPipelineFromSnapshot(
  snapshotDirectory: string,
  options: StableDiffusionPipelineLoadOptions = {},
): Promise<StableDiffusionPipelineBundle> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  assertUnsupportedSafetyCheckerAbsent(manifest);
  const configs = await loadStableDiffusionComponentConfigs(manifest);
  let vae: StableDiffusionAutoencoderKL | undefined;
  let unet: StableDiffusionUNet2DConditionModel | undefined;
  try {
    vae = await loadStableDiffusionAutoencoderFromSnapshot(manifest, options.autoencoder);
    unet = await loadStableDiffusionUNetFromSnapshot(manifest, options.unet);
    return new LoadedStableDiffusionPipelineBundle(
      snapshotDirectory,
      manifest,
      configs,
      vae,
      unet,
      createStableDiffusionScheduler(manifest.schedulerConfig),
    );
  } catch (error) {
    unet?.[Symbol.dispose]();
    vae?.[Symbol.dispose]();
    throw error;
  }
}
