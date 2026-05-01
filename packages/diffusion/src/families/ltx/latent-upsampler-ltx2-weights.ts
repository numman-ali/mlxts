import {
  contiguous,
  iterateSafetensors,
  MxArray,
  mxEval,
  type ParameterTree,
  transpose,
  treeFlatten,
} from "@mlxts/core";
import { readdirSync, statSync } from "fs";
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
import { Ltx2LatentUpsamplerModel, parseLtx2LatentUpsamplerConfig } from "./latent-upsampler-ltx2";

export type Ltx2LatentUpsamplerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-2 latent upsampler weights. */
export type Ltx2LatentUpsamplerWeightLoadResult = {
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

function pathIfFile(path: string): string | undefined {
  try {
    if (statSync(path).isFile()) {
      return path;
    }
  } catch {}
  return undefined;
}

function nextNode(current: unknown, segment: string, path: string): unknown {
  if (Array.isArray(current)) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0 || index >= current.length) {
      throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" has invalid index.`);
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" hit a non-object.`);
  }
  return Reflect.get(current, segment);
}

function assignLtx2LatentUpsamplerWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" is malformed.`);
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" has no object leaf.`);
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" is missing a leaf.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(`assignLtx2LatentUpsamplerWeightPath: "${path}" is not a tensor slot.`);
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

function ltx2LatentUpsamplerComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "latent_upsampler" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      "LTX-2 latent upsampler snapshot manifest is missing an enabled latent_upsampler.",
    );
  }
  return component;
}

function listWeightPaths(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true })
    .map((entry) => join(directory, entry.name))
    .filter(
      (path) =>
        pathIfFile(path) !== undefined &&
        (path.endsWith(".safetensors") || path.endsWith(".safetensors.index.json")),
    )
    .sort((left, right) => left.localeCompare(right));
}

function componentFromDirectory(directory: string): DiffusionSnapshotComponent {
  const configPath = pathIfFile(join(directory, "config.json"));
  return {
    name: "latent_upsampler",
    role: "latent-upsampler",
    library: "ltx2",
    className: "LTX2LatentUpsamplerModel",
    enabled: true,
    optional: false,
    subfolder: "latent_upsampler",
    directory,
    metadataPaths: configPath === undefined ? [] : [configPath],
    weightPaths: listWeightPaths(directory),
  };
}

async function loadComponentConfig(component: DiffusionSnapshotComponent): Promise<unknown> {
  const configPath = component.metadataPaths.find((path) => path.endsWith("config.json"));
  if (configPath === undefined) {
    throw new DiffusionConfigError(`${component.name} is missing config.json metadata.`);
  }
  try {
    return await Bun.file(configPath).json();
  } catch {
    throw new DiffusionConfigError(`${configPath} must contain valid JSON.`);
  }
}

/** Map a Diffusers LTX-2 latent upsampler tensor name onto the package module tree. */
export function ltx2LatentUpsamplerWeightPath(checkpointName: string): string | null {
  if (
    checkpointName.trim() === "" ||
    checkpointName.includes("num_batches_tracked") ||
    checkpointName === "upsampler.blur_down.kernel"
  ) {
    return null;
  }
  if (checkpointName.startsWith("initial_conv.")) {
    return checkpointName.replace("initial_conv.", "initialConv.");
  }
  if (checkpointName.startsWith("initial_norm.")) {
    return checkpointName.replace("initial_norm.", "initialNorm.");
  }
  if (checkpointName.startsWith("res_blocks.")) {
    return checkpointName.replace("res_blocks.", "resBlocks.");
  }
  if (checkpointName.startsWith("upsampler.0.")) {
    return checkpointName.replace("upsampler.0.", "upsampler.conv.");
  }
  if (checkpointName.startsWith("upsampler.conv.")) {
    return checkpointName;
  }
  if (checkpointName.startsWith("post_upsample_res_blocks.")) {
    return checkpointName.replace("post_upsample_res_blocks.", "postUpsampleResBlocks.");
  }
  if (checkpointName.startsWith("final_conv.")) {
    return checkpointName.replace("final_conv.", "finalConv.");
  }
  return null;
}

function isIgnorableLtx2LatentUpsamplerWeight(checkpointName: string): boolean {
  return (
    checkpointName.trim() === "" ||
    checkpointName.includes("num_batches_tracked") ||
    checkpointName === "upsampler.blur_down.kernel"
  );
}

/** Transform Diffusers LTX-2 upsampler tensors into MLX channel-last kernels. */
export function transformLtx2LatentUpsamplerWeight(_weightPath: string, tensor: MxArray): MxArray {
  if (tensor.shape.length === 5) {
    using transposed = transpose(tensor, [0, 2, 3, 4, 1]);
    return contiguous(transposed);
  }
  if (tensor.shape.length === 4) {
    using transposed = transpose(tensor, [0, 2, 3, 1]);
    return contiguous(transposed);
  }
  return tensor;
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
  options: Ltx2LatentUpsamplerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtx2LatentUpsamplerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function assignWeightTensor(
  model: Ltx2LatentUpsamplerModel,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtx2LatentUpsamplerWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtx2LatentUpsamplerWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: Ltx2LatentUpsamplerModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltx2LatentUpsamplerWeightPath(checkpointName);
  if (path === null) {
    if (!isIgnorableLtx2LatentUpsamplerWeight(checkpointName)) {
      unexpectedWeights.push(checkpointName);
    }
    tensor.free();
    return;
  }
  if (!expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into an LTX-2 latent upsampler module. */
export async function loadLtx2LatentUpsamplerWeights(
  model: Ltx2LatentUpsamplerModel,
  component: DiffusionSnapshotComponent,
  options: Ltx2LatentUpsamplerWeightLoadOptions = {},
): Promise<Ltx2LatentUpsamplerWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      consumeCheckpointTensor(model, expectedPaths, assignedPaths, unexpectedWeights, name, tensor);
    }
  }

  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(unexpectedWeights, options);

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Construct and load a standalone LTX-2 latent upsampler directory. */
export async function loadLtx2LatentUpsamplerFromDirectory(
  directory: string,
  options: Ltx2LatentUpsamplerWeightLoadOptions = {},
): Promise<Ltx2LatentUpsamplerModel> {
  const component = componentFromDirectory(directory);
  const model = new Ltx2LatentUpsamplerModel(
    parseLtx2LatentUpsamplerConfig(await loadComponentConfig(component)),
  );
  try {
    await loadLtx2LatentUpsamplerWeights(model, component, options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

/** Construct and load an LTX-2 latent upsampler from a sidecar snapshot manifest. */
export async function loadLtx2LatentUpsamplerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Ltx2LatentUpsamplerWeightLoadOptions = {},
): Promise<Ltx2LatentUpsamplerModel> {
  if (manifest.modelIndex.kind !== "ltx2-latent-upsample") {
    throw new DiffusionConfigError(
      "loadLtx2LatentUpsamplerFromSnapshot requires LTX2LatentUpsamplePipeline.",
    );
  }
  const component = ltx2LatentUpsamplerComponent(manifest);
  const model = new Ltx2LatentUpsamplerModel(
    parseLtx2LatentUpsamplerConfig(await loadComponentConfig(component)),
  );
  try {
    await loadLtx2LatentUpsamplerWeights(model, component, options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
