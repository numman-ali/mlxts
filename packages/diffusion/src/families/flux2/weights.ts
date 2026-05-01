import { iterateSafetensors, MxArray, mxEval, type ParameterTree, treeFlatten } from "@mlxts/core";
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
import {
  stableDiffusionAutoencoderWeightPath,
  transformStableDiffusionAutoencoderWeight,
} from "../stable-diffusion/weights";
import { Flux2KleinAutoencoderKL } from "./autoencoder";
import { loadFlux2KleinComponentConfigs } from "./config";

export type Flux2KleinAutoencoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading FLUX.2 Klein VAE weights. */
export type Flux2KleinAutoencoderWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const BATCH_NORM_MEAN_PATH = "bn.running_mean";
const BATCH_NORM_VAR_PATH = "bn.running_var";

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
        `assignFlux2KleinAutoencoderWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignFlux2KleinAutoencoderWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignFlux2KleinAutoencoderWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignFlux2KleinAutoencoderWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignFlux2KleinAutoencoderWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignFlux2KleinAutoencoderWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignFlux2KleinAutoencoderWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function flux2KleinAutoencoderComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "vae" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("FLUX.2 Klein snapshot manifest is missing an enabled VAE.");
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
  options: Flux2KleinAutoencoderWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadFlux2KleinAutoencoderWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

/** Map a Diffusers FLUX.2 VAE tensor name onto the package AutoencoderKLFlux2 tree. */
export function flux2KleinAutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName === "bn.num_batches_tracked") {
    return null;
  }
  if (checkpointName === BATCH_NORM_MEAN_PATH || checkpointName === BATCH_NORM_VAR_PATH) {
    return checkpointName;
  }
  return stableDiffusionAutoencoderWeightPath(checkpointName);
}

function isBatchNormStatPath(path: string): boolean {
  return path === BATCH_NORM_MEAN_PATH || path === BATCH_NORM_VAR_PATH;
}

function statValues(path: string, tensor: MxArray, expectedLength: number): number[] {
  if (!sameShape(tensor.shape, [expectedLength])) {
    throw new DiffusionWeightMismatchError(path, [expectedLength], tensor.shape);
  }
  return Array.from(tensor.toTypedArray(), Number);
}

function transformedWeight(checkpointName: string, weightPath: string, tensor: MxArray): MxArray {
  const transformed = transformStableDiffusionAutoencoderWeight(checkpointName, weightPath, tensor);
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

type BatchNormStatsLoadState = {
  mean: number[] | null;
  variance: number[] | null;
};

type AutoencoderWeightLoadState = {
  expectedPaths: ReadonlySet<string>;
  assignedPaths: Set<string>;
  unexpectedWeights: string[];
  batchNorm: BatchNormStatsLoadState;
};

function createWeightLoadState(model: Flux2KleinAutoencoderKL): AutoencoderWeightLoadState {
  return {
    expectedPaths: new Set([
      ...listParameterPaths(model.parameters()),
      BATCH_NORM_MEAN_PATH,
      BATCH_NORM_VAR_PATH,
    ]),
    assignedPaths: new Set<string>(),
    unexpectedWeights: [],
    batchNorm: {
      mean: null,
      variance: null,
    },
  };
}

function consumeBatchNormStat(
  model: Flux2KleinAutoencoderKL,
  state: AutoencoderWeightLoadState,
  path: string,
  tensor: MxArray,
): void {
  try {
    const values = statValues(path, tensor, model.packedLatentChannels);
    if (path === BATCH_NORM_MEAN_PATH) {
      state.batchNorm.mean = values;
    } else {
      state.batchNorm.variance = values;
    }
    state.assignedPaths.add(path);
  } finally {
    tensor.free();
  }
}

function consumeParameterTensor(
  model: Flux2KleinAutoencoderKL,
  state: AutoencoderWeightLoadState,
  checkpointName: string,
  path: string,
  tensor: MxArray,
): void {
  let assignedTensor = tensor;
  try {
    assignedTensor = transformedWeight(checkpointName, path, assignedTensor);
    assignFlux2KleinAutoencoderWeightPath(model, path, assignedTensor);
    state.assignedPaths.add(path);
  } catch (error) {
    assignedTensor.free();
    throw error;
  }
}

function consumeCheckpointTensor(
  model: Flux2KleinAutoencoderKL,
  state: AutoencoderWeightLoadState,
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = flux2KleinAutoencoderWeightPath(checkpointName);
  if (path === null) {
    tensor.free();
    return;
  }
  if (!state.expectedPaths.has(path)) {
    state.unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  if (isBatchNormStatPath(path)) {
    consumeBatchNormStat(model, state, path, tensor);
    return;
  }
  consumeParameterTensor(model, state, checkpointName, path, tensor);
}

/** Load Diffusers safetensors weights into a FLUX.2 Klein AutoencoderKLFlux2 module. */
export async function loadFlux2KleinAutoencoderWeights(
  model: Flux2KleinAutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: Flux2KleinAutoencoderWeightLoadOptions = {},
): Promise<Flux2KleinAutoencoderWeightLoadResult> {
  const state = createWeightLoadState(model);
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      consumeCheckpointTensor(model, state, name, tensor);
    }
  }

  throwIfMissingWeights(state.expectedPaths, state.assignedPaths);
  if (state.batchNorm.mean === null || state.batchNorm.variance === null) {
    throw new DiffusionMissingWeightsError([BATCH_NORM_MEAN_PATH, BATCH_NORM_VAR_PATH]);
  }
  model.setBatchNormStats(state.batchNorm.mean, state.batchNorm.variance);
  throwIfUnexpectedWeights(state.unexpectedWeights, options);

  return {
    assignedPaths: [...state.assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...state.unexpectedWeights].toSorted((left, right) =>
      left.localeCompare(right),
    ),
    shardCount: shardPaths.length,
  };
}

/** Construct and load the FLUX.2 Klein VAE component from a snapshot manifest. */
export async function loadFlux2KleinAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Flux2KleinAutoencoderWeightLoadOptions = {},
): Promise<Flux2KleinAutoencoderKL> {
  const configs = await loadFlux2KleinComponentConfigs(manifest);
  const model = new Flux2KleinAutoencoderKL(configs.vae);
  try {
    await loadFlux2KleinAutoencoderWeights(
      model,
      flux2KleinAutoencoderComponent(manifest),
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
