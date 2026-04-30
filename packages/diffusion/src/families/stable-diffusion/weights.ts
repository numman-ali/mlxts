/**
 * Stable Diffusion checkpoint weight mapping and component loading.
 * @module
 */

import {
  contiguous,
  iterateSafetensors,
  MxArray,
  mxEval,
  type ParameterTree,
  transpose,
  treeFlatten,
} from "@mlxts/core";
import { dirname, join } from "path";

import {
  DiffusionConfigError,
  DiffusionMissingWeightsError,
  DiffusionWeightMismatchError,
} from "../../errors";
import type {
  DiffusionSnapshotComponent,
  DiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import { StableDiffusionAutoencoderKL } from "./autoencoder";
import { loadStableDiffusionComponentConfigs } from "./config";
import { StableDiffusionUNet2DConditionModel } from "./unet";

export type StableDiffusionAutoencoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading Stable Diffusion VAE weights. */
export type StableDiffusionAutoencoderWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

export type StableDiffusionUNetWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading Stable Diffusion UNet weights. */
export type StableDiffusionUNetWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((dimension, index) => dimension === right[index]);
}

function formatParameterPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

function nextNode(current: unknown, segment: string, path: string): unknown {
  if (Array.isArray(current)) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0 || index >= current.length) {
      throw new Error(
        `assignDiffusionWeightPath: "${path}" segment "${segment}" is not a valid array index.`,
      );
    }
    return current[index];
  }

  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignDiffusionWeightPath: "${path}" cannot descend through a non-object segment.`,
    );
  }

  return Reflect.get(current, segment);
}

function assignDiffusionWeightPath(root: object, path: string, tensor: MxArray): void {
  if (path.trim() === "") {
    throw new Error("assignDiffusionWeightPath: path must not be empty.");
  }

  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignDiffusionWeightPath: "${path}" contains an undefined segment.`);
    }
    current = nextNode(current, segment, path);
  }

  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(`assignDiffusionWeightPath: "${path}" does not point to an object property.`);
  }

  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignDiffusionWeightPath: "${path}" is missing a leaf segment.`);
  }

  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(`assignDiffusionWeightPath: "${path}" does not point to an MxArray parameter.`);
  }

  if (!sameShape(existing.shape, tensor.shape)) {
    throw new DiffusionWeightMismatchError(path, existing.shape, tensor.shape);
  }

  existing.free();
  Reflect.set(current, leafKey, tensor);
}

function listParameterPaths(tree: ParameterTree): string[] {
  return treeFlatten(tree).map(([path]) => formatParameterPath(path));
}

function camelCaseVaeWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("down_blocks", "downBlocks")
    .replaceAll("up_blocks", "upBlocks")
    .replaceAll("conv_in", "convIn")
    .replaceAll("conv_norm_out", "convNormOut")
    .replaceAll("conv_out", "convOut")
    .replaceAll("conv_shortcut", "convShortcut")
    .replaceAll("post_quant_conv", "postQuantConv")
    .replaceAll("quant_conv", "quantConv")
    .replaceAll("downsamplers.0.conv", "downsample.conv")
    .replaceAll("upsamplers.0.conv", "upsample.conv")
    .replaceAll("mid_block.resnets.0", "midBlock.resnetIn")
    .replaceAll("mid_block.attentions.0", "midBlock.attention")
    .replaceAll("mid_block.resnets.1", "midBlock.resnetOut")
    .replaceAll("group_norm", "groupNorm")
    .replaceAll("to_q", "queryProjection")
    .replaceAll("to_k", "keyProjection")
    .replaceAll("to_v", "valueProjection")
    .replaceAll("to_out.0", "outputProjection");
}

/** Map a Diffusers VAE tensor name onto the package-owned AutoencoderKL parameter tree. */
export function stableDiffusionAutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseVaeWeightPath(checkpointName);
}

/** Transform a Diffusers VAE tensor into the package-owned parameter layout. */
export function transformStableDiffusionAutoencoderWeight(
  _checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  if (weightPath.endsWith(".weight") && tensor.shape.length === 4) {
    using transposed = transpose(tensor, [0, 2, 3, 1]);
    return contiguous(transposed);
  }
  return tensor;
}

function expectWeightMap(value: unknown, context: string): SafetensorsIndexWeightMap {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context}.weight_map must be a JSON object.`);
  }
  const output: SafetensorsIndexWeightMap = {};
  for (const [name, shard] of Object.entries(value)) {
    if (typeof shard !== "string" || shard.trim() === "") {
      throw new DiffusionConfigError(`${context}.weight_map.${name} must be a shard filename.`);
    }
    output[name] = shard;
  }
  return output;
}

async function loadSafetensorsIndex(indexPath: string): Promise<SafetensorsIndexWeightMap> {
  let rawIndex: unknown;
  try {
    rawIndex = await Bun.file(indexPath).json();
  } catch {
    throw new DiffusionConfigError(`safetensors index must contain valid JSON: ${indexPath}.`);
  }
  if (typeof rawIndex !== "object" || rawIndex === null || Array.isArray(rawIndex)) {
    throw new DiffusionConfigError(`safetensors index must be a JSON object: ${indexPath}.`);
  }
  return expectWeightMap(Reflect.get(rawIndex, "weight_map"), indexPath);
}

async function componentSafetensorShards(component: DiffusionSnapshotComponent): Promise<string[]> {
  const indexPaths = component.weightPaths.filter((path) =>
    path.endsWith(".safetensors.index.json"),
  );
  if (indexPaths.length > 1) {
    throw new DiffusionConfigError(`${component.name} has multiple safetensors index files.`);
  }

  const indexPath = indexPaths[0];
  if (indexPath !== undefined) {
    const weightMap = await loadSafetensorsIndex(indexPath);
    return [
      ...new Set(Object.values(weightMap).map((shard) => join(dirname(indexPath), shard))),
    ].sort((left, right) => left.localeCompare(right));
  }

  return component.weightPaths
    .filter((path) => path.endsWith(".safetensors"))
    .sort((left, right) => left.localeCompare(right));
}

function stableDiffusionVaeComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "vae" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("Stable Diffusion snapshot manifest is missing an enabled VAE.");
  }
  return component;
}

function stableDiffusionUNetComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "unet" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      "Stable Diffusion snapshot manifest is missing an enabled UNet.",
    );
  }
  return component;
}

function throwIfMissingWeights(
  expectedPaths: ReadonlySet<string>,
  assignedPaths: ReadonlySet<string>,
): void {
  const missingPaths = [...expectedPaths].filter((path) => !assignedPaths.has(path));
  if (missingPaths.length > 0) {
    throw new DiffusionMissingWeightsError(missingPaths);
  }
}

function throwIfUnexpectedWeights(
  unexpectedWeights: readonly string[],
  options: StableDiffusionAutoencoderWeightLoadOptions | StableDiffusionUNetWeightLoadOptions,
  owner: string,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `${owner}: checkpoint contained unexpected unmapped weights: ${[...unexpectedWeights]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function transformedWeight(checkpointName: string, weightPath: string, tensor: MxArray): MxArray {
  const transformed = transformStableDiffusionAutoencoderWeight(checkpointName, weightPath, tensor);
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

function camelCaseUNetWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("down_blocks", "downBlocks")
    .replaceAll("up_blocks", "upBlocks")
    .replaceAll("downsamplers.0.conv", "downsample")
    .replaceAll("upsamplers.0.conv", "upsample")
    .replaceAll("mid_block.resnets.0", "midBlock.resnetIn")
    .replaceAll("mid_block.attentions.0", "midBlock.attention")
    .replaceAll("mid_block.resnets.1", "midBlock.resnetOut")
    .replaceAll("time_embedding.linear_1", "timeEmbedding.linear1")
    .replaceAll("time_embedding.linear_2", "timeEmbedding.linear2")
    .replaceAll("add_embedding.linear_1", "addEmbedding.linear1")
    .replaceAll("add_embedding.linear_2", "addEmbedding.linear2")
    .replaceAll("conv_norm_out", "convNormOut")
    .replaceAll("conv_in", "convIn")
    .replaceAll("conv_out", "convOut")
    .replaceAll("time_emb_proj", "timeEmbeddingProjection")
    .replaceAll("conv_shortcut", "convShortcut")
    .replaceAll("transformer_blocks", "transformerBlocks")
    .replaceAll("attn1", "attention1")
    .replaceAll("attn2", "attention2")
    .replaceAll("to_q", "queryProjection")
    .replaceAll("to_k", "keyProjection")
    .replaceAll("to_v", "valueProjection")
    .replaceAll("to_out.0", "outputProjection")
    .replaceAll("ff.net.0.proj", "feedForward.projectionIn")
    .replaceAll("ff.net.2", "feedForward.projectionOut")
    .replaceAll("proj_in", "projectionIn")
    .replaceAll("proj_out", "projectionOut");
}

/** Map a Diffusers UNet tensor name onto the package-owned UNet parameter tree. */
export function stableDiffusionUNetWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseUNetWeightPath(checkpointName);
}

/** Transform a Diffusers UNet tensor into the package-owned parameter layout. */
export function transformStableDiffusionUNetWeight(
  _checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  if (weightPath.endsWith(".weight") && tensor.shape.length === 4) {
    using transposed = transpose(tensor, [0, 2, 3, 1]);
    return contiguous(transposed);
  }
  return tensor;
}

function transformedUNetWeight(
  checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  const transformed = transformStableDiffusionUNetWeight(checkpointName, weightPath, tensor);
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

/** Load Diffusers safetensors weights into a Stable Diffusion AutoencoderKL module. */
export async function loadStableDiffusionAutoencoderWeights(
  model: StableDiffusionAutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: StableDiffusionAutoencoderWeightLoadOptions = {},
): Promise<StableDiffusionAutoencoderWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      const path = stableDiffusionAutoencoderWeightPath(name);
      if (path === null || !expectedPaths.has(path)) {
        unexpectedWeights.push(name);
        tensor.free();
        continue;
      }

      let assignedTensor = tensor;
      try {
        assignedTensor = transformedWeight(name, path, assignedTensor);
        assignDiffusionWeightPath(model, path, assignedTensor);
        assignedPaths.add(path);
      } catch (error) {
        assignedTensor.free();
        throw error;
      }
    }
  }

  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(unexpectedWeights, options, "loadStableDiffusionAutoencoderWeights");

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Load Diffusers safetensors weights into a Stable Diffusion UNet2DConditionModel module. */
export async function loadStableDiffusionUNetWeights(
  model: StableDiffusionUNet2DConditionModel,
  component: DiffusionSnapshotComponent,
  options: StableDiffusionUNetWeightLoadOptions = {},
): Promise<StableDiffusionUNetWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      const path = stableDiffusionUNetWeightPath(name);
      if (path === null || !expectedPaths.has(path)) {
        unexpectedWeights.push(name);
        tensor.free();
        continue;
      }

      let assignedTensor = tensor;
      try {
        assignedTensor = transformedUNetWeight(name, path, assignedTensor);
        assignDiffusionWeightPath(model, path, assignedTensor);
        assignedPaths.add(path);
      } catch (error) {
        assignedTensor.free();
        throw error;
      }
    }
  }

  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(unexpectedWeights, options, "loadStableDiffusionUNetWeights");

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Construct and load the Stable Diffusion AutoencoderKL component from a snapshot manifest. */
export async function loadStableDiffusionAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: StableDiffusionAutoencoderWeightLoadOptions = {},
): Promise<StableDiffusionAutoencoderKL> {
  const configs = await loadStableDiffusionComponentConfigs(manifest);
  const model = new StableDiffusionAutoencoderKL(configs.vae);
  try {
    await loadStableDiffusionAutoencoderWeights(
      model,
      stableDiffusionVaeComponent(manifest),
      options,
    );
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

/** Construct and load the Stable Diffusion UNet component from a snapshot manifest. */
export async function loadStableDiffusionUNetFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: StableDiffusionUNetWeightLoadOptions = {},
): Promise<StableDiffusionUNet2DConditionModel> {
  const configs = await loadStableDiffusionComponentConfigs(manifest);
  const model = new StableDiffusionUNet2DConditionModel(configs.unet);
  try {
    await loadStableDiffusionUNetWeights(model, stableDiffusionUNetComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
