/**
 * FLUX.1 AutoencoderKL wrapper and checkpoint loading.
 * @module
 */

import { mxEval, treeFlatten } from "@mlxts/core";

import { DiffusionConfigError } from "../../errors";
import type {
  DiffusionSnapshotComponent,
  DiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import { StableDiffusionAutoencoderKL } from "../stable-diffusion/autoencoder";
import type { StableDiffusionAutoencoderConfig } from "../stable-diffusion/config";
import {
  loadStableDiffusionAutoencoderWeights,
  type StableDiffusionAutoencoderWeightLoadOptions,
  type StableDiffusionAutoencoderWeightLoadResult,
} from "../stable-diffusion/weights";
import { type FluxAutoencoderConfig, loadFluxComponentConfigs } from "./config";

export type FluxAutoencoderWeightLoadOptions = StableDiffusionAutoencoderWeightLoadOptions;

/** Assignment summary returned after loading FLUX VAE weights. */
export type FluxAutoencoderWeightLoadResult = StableDiffusionAutoencoderWeightLoadResult;

function stableConfigForFluxAutoencoder(
  config: FluxAutoencoderConfig,
): StableDiffusionAutoencoderConfig {
  return {
    inChannels: config.inChannels,
    outChannels: config.outChannels,
    latentChannels: config.latentChannels,
    latentChannelsOut: config.latentChannelsOut,
    useQuantConv: config.useQuantConv,
    usePostQuantConv: config.usePostQuantConv,
    blockOutChannels: config.blockOutChannels,
    layersPerBlock: config.layersPerBlock,
    normNumGroups: config.normNumGroups,
    scalingFactor: config.scalingFactor,
    downBlockTypes: config.downBlockTypes,
    upBlockTypes: config.upBlockTypes,
    forceUpcast: config.forceUpcast,
    rawConfig: config.rawConfig,
  };
}

function fluxAutoencoderComponent(manifest: DiffusionSnapshotManifest): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "vae" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("FLUX snapshot manifest is missing an enabled VAE.");
  }
  return component;
}

/** FLUX VAE decoder with shift-aware latent normalization metadata. */
export class FluxAutoencoderKL extends StableDiffusionAutoencoderKL {
  #shiftFactor: number;

  constructor(config: FluxAutoencoderConfig) {
    super(stableConfigForFluxAutoencoder(config));
    this.#shiftFactor = config.shiftFactor;
  }

  get shiftFactor(): number {
    return this.#shiftFactor;
  }
}

/** Load Diffusers safetensors weights into a FLUX AutoencoderKL module. */
export async function loadFluxAutoencoderWeights(
  model: FluxAutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: FluxAutoencoderWeightLoadOptions = {},
): Promise<FluxAutoencoderWeightLoadResult> {
  return loadStableDiffusionAutoencoderWeights(model, component, options);
}

/** Construct and load the FLUX VAE component from a snapshot manifest. */
export async function loadFluxAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: FluxAutoencoderWeightLoadOptions = {},
): Promise<FluxAutoencoderKL> {
  const configs = await loadFluxComponentConfigs(manifest);
  const model = new FluxAutoencoderKL(configs.vae);
  try {
    await loadFluxAutoencoderWeights(model, fluxAutoencoderComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
