/**
 * Z-Image checkpoint weight mapping and loading.
 * @module
 */

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
  FluxAutoencoderKL,
  type FluxAutoencoderWeightLoadOptions,
  loadFluxAutoencoderWeights,
} from "../flux/autoencoder";
import { loadZImageComponentConfigs } from "./config";
import { ZImageTransformer2DModel } from "./transformer";
import { type ZImageWeightPlan, zImageTransformerWeightPlan } from "./weight-mapping";

export { FluxAutoencoderKL as ZImageAutoencoderKL } from "../flux/autoencoder";
export { zImageTransformerWeightPath } from "./weight-mapping";

export type ZImageTransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading Z-Image transformer weights. */
export type ZImageTransformerWeightLoadResult = {
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
        `assignZImageWeightPath: "${path}" segment "${segment}" is not a valid array index.`,
      );
    }
    return current[index];
  }

  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignZImageWeightPath: "${path}" cannot descend through a non-object segment.`,
    );
  }

  return Reflect.get(current, segment);
}

function assignZImageWeightPath(root: object, path: string, tensor: MxArray): void {
  if (path.trim() === "") {
    throw new Error("assignZImageWeightPath: path must not be empty.");
  }

  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignZImageWeightPath: "${path}" contains an undefined segment.`);
    }
    current = nextNode(current, segment, path);
  }

  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(`assignZImageWeightPath: "${path}" does not point to an object property.`);
  }

  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignZImageWeightPath: "${path}" is missing a leaf segment.`);
  }

  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(`assignZImageWeightPath: "${path}" does not point to an MxArray parameter.`);
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

function zImageComponent(
  manifest: DiffusionSnapshotManifest,
  componentName: "transformer" | "vae",
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === componentName && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      `Z-Image snapshot manifest is missing an enabled ${componentName}.`,
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
  options: ZImageTransformerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadZImageTransformerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

/** Transform a Diffusers Z-Image transformer tensor into the package-owned parameter layout. */
export function transformZImageTransformerWeight(_weightPath: string, tensor: MxArray): MxArray {
  return tensor;
}

function assignDirectWeight(
  model: ZImageTransformer2DModel,
  plan: ZImageWeightPlan,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformedTensor = transformZImageTransformerWeight(plan.path, tensor);
    if (transformedTensor !== tensor) {
      tensor.free();
    }
    assignedTensor = transformedTensor;
    assignZImageWeightPath(model, plan.path, assignedTensor);
    assignedPaths.add(plan.path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function retainPlannedWeight(
  model: ZImageTransformer2DModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  name: string,
  tensor: MxArray,
): void {
  const plan = zImageTransformerWeightPlan(model.config, name);
  if (plan === null || !expectedPaths.has(plan.path)) {
    unexpectedWeights.push(name);
    tensor.free();
    return;
  }
  assignDirectWeight(model, plan, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into a Z-Image transformer module. */
export async function loadZImageTransformerWeights(
  model: ZImageTransformer2DModel,
  component: DiffusionSnapshotComponent,
  options: ZImageTransformerWeightLoadOptions = {},
): Promise<ZImageTransformerWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  for (const shardPath of shardPaths) {
    for await (const { name, tensor } of iterateSafetensors(shardPath)) {
      retainPlannedWeight(model, expectedPaths, assignedPaths, unexpectedWeights, name, tensor);
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

/** Construct and load the Z-Image transformer component from a snapshot manifest. */
export async function loadZImageTransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: ZImageTransformerWeightLoadOptions = {},
): Promise<ZImageTransformer2DModel> {
  const configs = await loadZImageComponentConfigs(manifest);
  const model = new ZImageTransformer2DModel(configs.transformer);
  try {
    await loadZImageTransformerWeights(model, zImageComponent(manifest, "transformer"), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

/** Construct and load the Z-Image VAE component from a snapshot manifest. */
export async function loadZImageAutoencoderFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: FluxAutoencoderWeightLoadOptions = {},
): Promise<FluxAutoencoderKL> {
  const configs = await loadZImageComponentConfigs(manifest);
  const model = new FluxAutoencoderKL(configs.vae);
  try {
    await loadFluxAutoencoderWeights(model, zImageComponent(manifest, "vae"), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
