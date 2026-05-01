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
import { StableDiffusion3AutoencoderKL } from "./autoencoder";
import { loadStableDiffusion3ComponentConfigs } from "./config";
import { StableDiffusion3Transformer2DModel } from "./transformer";
import {
  isIgnoredStableDiffusion3TransformerWeight,
  stableDiffusion3AutoencoderWeightPath,
  stableDiffusion3TransformerWeightPath,
} from "./weight-mapping";

export {
  stableDiffusion3AutoencoderWeightPath,
  stableDiffusion3TransformerWeightPath,
} from "./weight-mapping";

export type StableDiffusion3AutoencoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

export type StableDiffusion3TransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading Stable Diffusion 3 VAE weights. */
export type StableDiffusion3AutoencoderWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

/** Assignment summary returned after loading Stable Diffusion 3 transformer weights. */
export type StableDiffusion3TransformerWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length && left.every((dimension, index) => dimension === right[index])
  );
}

function formatParameterPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

function nextNode(current: unknown, segment: string, path: string): unknown {
  if (Array.isArray(current)) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0 || index >= current.length) {
      throw new Error(
        `assignStableDiffusion3WeightPath: "${path}" segment "${segment}" is not a valid array index.`,
      );
    }
    return current[index];
  }

  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignStableDiffusion3WeightPath: "${path}" cannot descend through a non-object segment.`,
    );
  }

  return Reflect.get(current, segment);
}

function assignStableDiffusion3WeightPath(root: object, path: string, tensor: MxArray): void {
  if (path.trim() === "") {
    throw new Error("assignStableDiffusion3WeightPath: path must not be empty.");
  }

  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignStableDiffusion3WeightPath: "${path}" contains an undefined segment.`);
    }
    current = nextNode(current, segment, path);
  }

  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignStableDiffusion3WeightPath: "${path}" does not point to an object property.`,
    );
  }

  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignStableDiffusion3WeightPath: "${path}" is missing a leaf segment.`);
  }

  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignStableDiffusion3WeightPath: "${path}" does not point to an MxArray parameter.`,
    );
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

function stableDiffusion3Component(
  manifest: DiffusionSnapshotManifest,
  componentName: "transformer" | "vae",
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === componentName && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      `Stable Diffusion 3 snapshot manifest is missing an enabled ${componentName}.`,
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
  options:
    | StableDiffusion3AutoencoderWeightLoadOptions
    | StableDiffusion3TransformerWeightLoadOptions,
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

/** Transform a Diffusers SD3 VAE tensor into the package-owned layout. */
export function transformStableDiffusion3AutoencoderWeight(
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

/** Transform a Diffusers SD3 transformer tensor into the package-owned layout. */
export function transformStableDiffusion3TransformerWeight(
  _checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  if (weightPath === "posEmbed.projection.weight" && tensor.shape.length === 4) {
    using transposed = transpose(tensor, [0, 2, 3, 1]);
    return contiguous(transposed);
  }
  return tensor;
}

function transformedAutoencoderWeight(
  checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  const transformed = transformStableDiffusion3AutoencoderWeight(
    checkpointName,
    weightPath,
    tensor,
  );
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

function transformedTransformerWeight(
  checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  const transformed = transformStableDiffusion3TransformerWeight(
    checkpointName,
    weightPath,
    tensor,
  );
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

/** Load Diffusers safetensors weights into a Stable Diffusion 3 AutoencoderKL module. */
export async function loadStableDiffusion3AutoencoderWeights(
  model: StableDiffusion3AutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: StableDiffusion3AutoencoderWeightLoadOptions = {},
): Promise<StableDiffusion3AutoencoderWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      const path = stableDiffusion3AutoencoderWeightPath(name);
      if (path === null || !expectedPaths.has(path)) {
        unexpectedWeights.push(name);
        tensor.free();
        continue;
      }

      let assignedTensor: MxArray | null = tensor;
      try {
        assignedTensor = transformedAutoencoderWeight(name, path, assignedTensor);
        assignStableDiffusion3WeightPath(model, path, assignedTensor);
        assignedPaths.add(path);
        assignedTensor = null;
      } finally {
        assignedTensor?.free();
      }
    }
  }

  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(unexpectedWeights, options, "loadStableDiffusion3AutoencoderWeights");

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Load Diffusers safetensors weights into a Stable Diffusion 3 transformer module. */
export async function loadStableDiffusion3TransformerWeights(
  model: StableDiffusion3Transformer2DModel,
  component: DiffusionSnapshotComponent,
  options: StableDiffusion3TransformerWeightLoadOptions = {},
): Promise<StableDiffusion3TransformerWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      if (isIgnoredStableDiffusion3TransformerWeight(name)) {
        tensor.free();
        continue;
      }
      const path = stableDiffusion3TransformerWeightPath(name);
      if (path === null || !expectedPaths.has(path)) {
        unexpectedWeights.push(name);
        tensor.free();
        continue;
      }

      let assignedTensor: MxArray | null = tensor;
      try {
        assignedTensor = transformedTransformerWeight(name, path, assignedTensor);
        assignStableDiffusion3WeightPath(model, path, assignedTensor);
        assignedPaths.add(path);
        assignedTensor = null;
      } finally {
        assignedTensor?.free();
      }
    }
  }

  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(unexpectedWeights, options, "loadStableDiffusion3TransformerWeights");

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Construct and load the SD3 VAE component from a snapshot manifest. */
export async function loadStableDiffusion3AutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: StableDiffusion3AutoencoderWeightLoadOptions = {},
): Promise<StableDiffusion3AutoencoderKL> {
  const configs = await loadStableDiffusion3ComponentConfigs(manifest);
  const model = new StableDiffusion3AutoencoderKL(configs.vae);
  try {
    await loadStableDiffusion3AutoencoderWeights(
      model,
      stableDiffusion3Component(manifest, "vae"),
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

/** Construct and load the SD3 transformer component from a snapshot manifest. */
export async function loadStableDiffusion3TransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: StableDiffusion3TransformerWeightLoadOptions = {},
): Promise<StableDiffusion3Transformer2DModel> {
  const configs = await loadStableDiffusion3ComponentConfigs(manifest);
  const model = new StableDiffusion3Transformer2DModel(configs.transformer);
  try {
    await loadStableDiffusion3TransformerWeights(
      model,
      stableDiffusion3Component(manifest, "transformer"),
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
