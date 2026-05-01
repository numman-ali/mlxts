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
import { Ltx2VideoAutoencoderKL } from "./autoencoder-ltx2";
import { loadLtxComponentConfigs } from "./config";

export type Ltx2VideoAutoencoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-2 video VAE decoder weights. */
export type Ltx2VideoAutoencoderWeightLoadResult = {
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
        `assignLtx2VideoAutoencoderWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignLtx2VideoAutoencoderWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignLtx2VideoAutoencoderWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignLtx2VideoAutoencoderWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignLtx2VideoAutoencoderWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtx2VideoAutoencoderWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignLtx2VideoAutoencoderWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function ltx2VideoAutoencoderComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "vae" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("LTX-2 snapshot manifest is missing an enabled video VAE.");
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
    .replaceAll("norm_out", "normOut")
    .replaceAll("conv_shortcut", "convShortcut")
    .replaceAll("per_channel_scale", "perChannelScale")
    .replaceAll("scale_shift_table", "scaleShiftTable")
    .replaceAll("time_embedder", "timeEmbedder");
}

/** Map a Diffusers LTX-2 video VAE tensor name onto the package decoder tree. */
export function ltx2VideoAutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName === LATENTS_MEAN_PATH || checkpointName === LATENTS_STD_PATH) {
    return checkpointName;
  }
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseDecoderWeightPath(checkpointName);
}

/** Transform Diffusers LTX-2 video VAE tensors into package-owned parameter layout. */
export function transformLtx2VideoAutoencoderWeight(_weightPath: string, tensor: MxArray): MxArray {
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
  options: Ltx2VideoAutoencoderWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtx2VideoAutoencoderWeights: checkpoint contained unexpected unmapped weights: ${[
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
  model: Ltx2VideoAutoencoderKL,
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
  model: Ltx2VideoAutoencoderKL,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtx2VideoAutoencoderWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtx2VideoAutoencoderWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: Ltx2VideoAutoencoderKL,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  stats: LatentStatsState,
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltx2VideoAutoencoderWeightPath(checkpointName);
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

/** Load Diffusers safetensors weights into an LTX-2 video decoder-only VAE module. */
export async function loadLtx2VideoAutoencoderWeights(
  model: Ltx2VideoAutoencoderKL,
  component: DiffusionSnapshotComponent,
  options: Ltx2VideoAutoencoderWeightLoadOptions = {},
): Promise<Ltx2VideoAutoencoderWeightLoadResult> {
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

/** Construct and load the LTX-2 video decoder-only VAE from a snapshot manifest. */
export async function loadLtx2VideoAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Ltx2VideoAutoencoderWeightLoadOptions = {},
): Promise<Ltx2VideoAutoencoderKL> {
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx2") {
    throw new DiffusionConfigError("loadLtx2VideoAutoencoderFromSnapshot requires LTX2Pipeline.");
  }
  const model = new Ltx2VideoAutoencoderKL(configs.vae);
  try {
    await loadLtx2VideoAutoencoderWeights(model, ltx2VideoAutoencoderComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
