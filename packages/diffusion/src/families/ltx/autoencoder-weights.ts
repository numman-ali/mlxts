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
import { LtxVideoAutoencoderKL } from "./autoencoder";
import { loadLtxComponentConfigs } from "./config";

export type LtxVideoAutoencoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-Video VAE decoder weights. */
export type LtxVideoAutoencoderWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const LATENTS_MEAN_PATH = "latents_mean";
const LATENTS_STD_PATH = "latents_std";

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
        `assignLtxVideoAutoencoderWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignLtxVideoAutoencoderWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignLtxVideoAutoencoderWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignLtxVideoAutoencoderWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignLtxVideoAutoencoderWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtxVideoAutoencoderWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignLtxVideoAutoencoderWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function ltxVideoAutoencoderComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "vae" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("LTX-Video snapshot manifest is missing an enabled VAE.");
  }
  return component;
}

function camelCaseDecoderWeightPath(checkpointName: string): string | null {
  if (!checkpointName.startsWith("decoder.")) {
    return null;
  }
  return checkpointName
    .replaceAll("conv_in", "convIn")
    .replaceAll("conv_out", "convOut")
    .replaceAll("mid_block", "midBlock")
    .replaceAll("up_blocks", "upBlocks")
    .replaceAll("upsamplers.0", "upsampler")
    .replaceAll("norm_out", "normOut")
    .replaceAll("conv_shortcut", "convShortcut");
}

/** Map a Diffusers LTX-Video VAE tensor name onto the package decoder tree. */
export function ltxVideoAutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName === LATENTS_MEAN_PATH || checkpointName === LATENTS_STD_PATH) {
    return checkpointName;
  }
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseDecoderWeightPath(checkpointName);
}

/** Transform Diffusers LTX-Video VAE tensors into package-owned parameter layout. */
export function transformLtxVideoAutoencoderWeight(_weightPath: string, tensor: MxArray): MxArray {
  if (tensor.shape.length === 5) {
    using transposed = transpose(tensor, [0, 2, 3, 4, 1]);
    return contiguous(transposed);
  }
  return tensor;
}

function statValues(path: string, tensor: MxArray, expectedLength: number): number[] {
  if (!sameShape(tensor.shape, [expectedLength])) {
    throw new DiffusionWeightMismatchError(path, [expectedLength], tensor.shape);
  }
  return Array.from(tensor.toTypedArray(), Number);
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
  options: LtxVideoAutoencoderWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtxVideoAutoencoderWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

type LatentStatsState = {
  mean: number[] | null;
  std: number[] | null;
};

function consumeLatentStat(
  state: LatentStatsState,
  model: LtxVideoAutoencoderKL,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  try {
    const values = statValues(path, tensor, model.latentChannels);
    if (path === LATENTS_MEAN_PATH) {
      state.mean = values;
    } else {
      state.std = values;
    }
    assignedPaths.add(path);
  } finally {
    tensor.free();
  }
}

function assignWeightTensor(
  model: LtxVideoAutoencoderKL,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtxVideoAutoencoderWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtxVideoAutoencoderWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: LtxVideoAutoencoderKL,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  stats: LatentStatsState,
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltxVideoAutoencoderWeightPath(checkpointName);
  if (path === LATENTS_MEAN_PATH || path === LATENTS_STD_PATH) {
    consumeLatentStat(stats, model, path, tensor, assignedPaths);
    return;
  }
  if (path === null || !expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into an LTX-Video decoder-only VAE module. */
export async function loadLtxVideoAutoencoderWeights(
  model: LtxVideoAutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: LtxVideoAutoencoderWeightLoadOptions = {},
): Promise<LtxVideoAutoencoderWeightLoadResult> {
  const expectedPaths = new Set([
    ...listParameterPaths(model.parameters()),
    LATENTS_MEAN_PATH,
    LATENTS_STD_PATH,
  ]);
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const stats: LatentStatsState = { mean: null, std: null };
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      consumeCheckpointTensor(
        model,
        expectedPaths,
        assignedPaths,
        unexpectedWeights,
        stats,
        name,
        tensor,
      );
    }
  }

  if (stats.mean === null || stats.std === null) {
    throwIfMissingWeights(expectedPaths, assignedPaths);
  } else {
    model.setLatentStats(stats.mean, stats.std);
    throwIfMissingWeights(expectedPaths, assignedPaths);
  }
  throwIfUnexpectedWeights(unexpectedWeights, options);

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Construct and load the classic LTX-Video decoder-only VAE from a snapshot manifest. */
export async function loadLtxVideoAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: LtxVideoAutoencoderWeightLoadOptions = {},
): Promise<LtxVideoAutoencoderKL> {
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx-video") {
    throw new DiffusionConfigError("loadLtxVideoAutoencoderFromSnapshot requires LTX-Video.");
  }
  const model = new LtxVideoAutoencoderKL(configs.vae);
  try {
    await loadLtxVideoAutoencoderWeights(model, ltxVideoAutoencoderComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
