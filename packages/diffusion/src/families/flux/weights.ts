/**
 * FLUX.1 transformer checkpoint weight mapping and loading.
 * @module
 */

import {
  concatenate,
  iterateSafetensors,
  MxArray,
  mxEval,
  type ParameterTree,
  split,
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
import { loadFluxComponentConfigs } from "./config";
import { FluxTransformer2DModel } from "./transformer";
import type { DirectWeightPlan, FusedWeightPlan } from "./weight-mapping";
import { fluxTransformerWeightPlan } from "./weight-mapping";

export { fluxTransformerWeightPath } from "./weight-mapping";

export type FluxTransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading FLUX transformer weights. */
export type FluxTransformerWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

type PendingFusedWeight = {
  path: string;
  parts: (MxArray | null)[];
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function freeParts(parts: readonly MxArray[]): void {
  for (const part of parts) {
    part.free();
  }
}

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
        `assignFluxWeightPath: "${path}" segment "${segment}" is not a valid array index.`,
      );
    }
    return current[index];
  }

  if (typeof current !== "object" || current === null) {
    throw new Error(`assignFluxWeightPath: "${path}" cannot descend through a non-object segment.`);
  }

  return Reflect.get(current, segment);
}

function assignFluxWeightPath(root: object, path: string, tensor: MxArray): void {
  if (path.trim() === "") {
    throw new Error("assignFluxWeightPath: path must not be empty.");
  }

  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignFluxWeightPath: "${path}" contains an undefined segment.`);
    }
    current = nextNode(current, segment, path);
  }

  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(`assignFluxWeightPath: "${path}" does not point to an object property.`);
  }

  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignFluxWeightPath: "${path}" is missing a leaf segment.`);
  }

  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(`assignFluxWeightPath: "${path}" does not point to an MxArray parameter.`);
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

function fluxTransformerComponent(manifest: DiffusionSnapshotManifest): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "transformer" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("FLUX snapshot manifest is missing an enabled transformer.");
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
  options: FluxTransformerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadFluxTransformerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function pendingFusedWeight(
  pendingWeights: Map<string, PendingFusedWeight>,
  plan: FusedWeightPlan,
): PendingFusedWeight {
  const existing = pendingWeights.get(plan.path);
  if (existing !== undefined) {
    return existing;
  }
  const pending = {
    path: plan.path,
    parts: Array.from({ length: plan.partCount }, () => null),
  };
  pendingWeights.set(plan.path, pending);
  return pending;
}

function retainFusedPart(
  pendingWeights: Map<string, PendingFusedWeight>,
  plan: FusedWeightPlan,
  tensor: MxArray,
): void {
  const pending = pendingFusedWeight(pendingWeights, plan);
  const existing = pending.parts[plan.partIndex];
  if (existing !== null) {
    throw new DiffusionConfigError(`duplicate FLUX fused checkpoint part for ${plan.path}.`);
  }
  pending.parts[plan.partIndex] = tensor;
}

function completeFusedParts(pending: PendingFusedWeight): MxArray[] | null {
  const parts: MxArray[] = [];
  for (const part of pending.parts) {
    if (part === null) {
      return null;
    }
    parts.push(part);
  }
  return parts;
}

function freePendingFusedWeights(pendingWeights: Map<string, PendingFusedWeight>): void {
  for (const pending of pendingWeights.values()) {
    for (const part of pending.parts) {
      part?.free();
    }
  }
  pendingWeights.clear();
}

function swapFinalModulationHalves(tensor: MxArray): MxArray {
  const parts = split(tensor, 2, 0);
  try {
    const scale = partAt(parts, 0, "swapFinalModulationHalves");
    const shift = partAt(parts, 1, "swapFinalModulationHalves");
    return concatenate([shift, scale], 0);
  } finally {
    freeParts(parts);
  }
}

/** Transform a Diffusers FLUX transformer tensor into the package-owned parameter layout. */
export function transformFluxTransformerWeight(weightPath: string, tensor: MxArray): MxArray {
  if (
    weightPath === "finalLayer.modulation.weight" ||
    weightPath === "finalLayer.modulation.bias"
  ) {
    return swapFinalModulationHalves(tensor);
  }
  return tensor;
}

function assignFusedWeights(
  model: FluxTransformer2DModel,
  pendingWeights: Map<string, PendingFusedWeight>,
  assignedPaths: Set<string>,
): void {
  for (const pending of pendingWeights.values()) {
    const parts = completeFusedParts(pending);
    if (parts === null) {
      continue;
    }
    let fusedTensor: MxArray | null = concatenate(parts, 0);
    try {
      assignFluxWeightPath(model, pending.path, fusedTensor);
      assignedPaths.add(pending.path);
      fusedTensor = null;
    } finally {
      fusedTensor?.free();
    }
  }
}

function assignDirectWeight(
  model: FluxTransformer2DModel,
  plan: DirectWeightPlan,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  const sourceTensor = tensor;
  let assignedTensor: MxArray | null = sourceTensor;
  try {
    const transformedTensor = transformFluxTransformerWeight(plan.path, sourceTensor);
    if (transformedTensor !== sourceTensor) {
      sourceTensor.free();
    }
    assignedTensor = transformedTensor;
    assignFluxWeightPath(model, plan.path, assignedTensor);
    assignedPaths.add(plan.path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function retainPlannedWeight(
  model: FluxTransformer2DModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  pendingFusedWeights: Map<string, PendingFusedWeight>,
  name: string,
  tensor: MxArray,
): void {
  const plan = fluxTransformerWeightPlan(name);
  if (plan === null || !expectedPaths.has(plan.path)) {
    unexpectedWeights.push(name);
    tensor.free();
    return;
  }

  if (plan.kind === "fused") {
    try {
      retainFusedPart(pendingFusedWeights, plan, tensor);
    } catch (error) {
      tensor.free();
      throw error;
    }
    return;
  }

  assignDirectWeight(model, plan, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into a FLUX transformer module. */
export async function loadFluxTransformerWeights(
  model: FluxTransformer2DModel,
  component: DiffusionSnapshotComponent,
  options: FluxTransformerWeightLoadOptions = {},
): Promise<FluxTransformerWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const pendingFusedWeights = new Map<string, PendingFusedWeight>();
  const shardPaths = await componentSafetensorShards(component);

  if (shardPaths.length === 0) {
    throw new DiffusionConfigError(`${component.name} has no safetensors weight shards.`);
  }

  try {
    for (const shardPath of shardPaths) {
      for await (const { name, tensor } of iterateSafetensors(shardPath)) {
        retainPlannedWeight(
          model,
          expectedPaths,
          assignedPaths,
          unexpectedWeights,
          pendingFusedWeights,
          name,
          tensor,
        );
      }
    }

    assignFusedWeights(model, pendingFusedWeights, assignedPaths);
    throwIfMissingWeights(expectedPaths, assignedPaths);
    throwIfUnexpectedWeights(unexpectedWeights, options);
  } finally {
    freePendingFusedWeights(pendingFusedWeights);
  }

  return {
    assignedPaths: [...assignedPaths].toSorted((left, right) => left.localeCompare(right)),
    unexpectedWeights: [...unexpectedWeights].toSorted((left, right) => left.localeCompare(right)),
    shardCount: shardPaths.length,
  };
}

/** Construct and load the FLUX transformer component from a snapshot manifest. */
export async function loadFluxTransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: FluxTransformerWeightLoadOptions = {},
): Promise<FluxTransformer2DModel> {
  const configs = await loadFluxComponentConfigs(manifest);
  const model = new FluxTransformer2DModel(configs.transformer);
  try {
    await loadFluxTransformerWeights(model, fluxTransformerComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
