/**
 * FLUX.2 Klein transformer checkpoint weight loading.
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
import { loadFlux2KleinComponentConfigs } from "./config";
import { Flux2KleinTransformer2DModel } from "./transformer";

export type Flux2KleinTransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading FLUX.2 Klein transformer weights. */
export type Flux2KleinTransformerWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  [
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "timeGuidanceEmbed.timestepEmbedder.linear1.weight",
  ],
  [
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "timeGuidanceEmbed.timestepEmbedder.linear2.weight",
  ],
  [
    "time_guidance_embed.guidance_embedder.linear_1.weight",
    "timeGuidanceEmbed.guidanceEmbedder.linear1.weight",
  ],
  [
    "time_guidance_embed.guidance_embedder.linear_2.weight",
    "timeGuidanceEmbed.guidanceEmbedder.linear2.weight",
  ],
  ["double_stream_modulation_img.linear.weight", "doubleStreamModulationImg.linear.weight"],
  ["double_stream_modulation_txt.linear.weight", "doubleStreamModulationTxt.linear.weight"],
  ["single_stream_modulation.linear.weight", "singleStreamModulation.linear.weight"],
  ["x_embedder.weight", "xEmbedder.weight"],
  ["context_embedder.weight", "contextEmbedder.weight"],
  ["norm_out.linear.weight", "normOut.linear.weight"],
  ["proj_out.weight", "projOut.weight"],
]);

const DOUBLE_BLOCK_WEIGHT_PATHS = new Map<string, string>([
  ["attn.to_q", "attn.toQ"],
  ["attn.to_k", "attn.toK"],
  ["attn.to_v", "attn.toV"],
  ["attn.add_q_proj", "attn.addQProj"],
  ["attn.add_k_proj", "attn.addKProj"],
  ["attn.add_v_proj", "attn.addVProj"],
  ["attn.norm_q", "attn.norm.queryNorm"],
  ["attn.norm_k", "attn.norm.keyNorm"],
  ["attn.norm_added_q", "attn.addedNorm.queryNorm"],
  ["attn.norm_added_k", "attn.addedNorm.keyNorm"],
  ["attn.to_out.0", "attn.toOut"],
  ["attn.to_add_out", "attn.toAddOut"],
  ["ff.linear_in", "ff.linearIn"],
  ["ff.linear_out", "ff.linearOut"],
  ["ff_context.linear_in", "ffContext.linearIn"],
  ["ff_context.linear_out", "ffContext.linearOut"],
]);

const SINGLE_BLOCK_WEIGHT_PATHS = new Map<string, string>([
  ["attn.to_qkv_mlp_proj", "attn.toQkvMlpProj"],
  ["attn.norm_q", "attn.norm.queryNorm"],
  ["attn.norm_k", "attn.norm.keyNorm"],
  ["attn.to_out", "attn.toOut"],
]);

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
        `assignFlux2KleinTransformerWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignFlux2KleinTransformerWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignFlux2KleinTransformerWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignFlux2KleinTransformerWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignFlux2KleinTransformerWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignFlux2KleinTransformerWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignFlux2KleinTransformerWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function flux2KleinTransformerComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "transformer" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      "FLUX.2 Klein snapshot manifest is missing an enabled transformer.",
    );
  }
  return component;
}

function doubleBlockWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.(.+)\.weight$/);
  if (match === null) {
    return null;
  }
  const [, index, local] = match;
  if (index === undefined || local === undefined) {
    return null;
  }
  const path = DOUBLE_BLOCK_WEIGHT_PATHS.get(local);
  return path === undefined ? null : `transformerBlocks.${index}.${path}.weight`;
}

function singleBlockWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^single_transformer_blocks\.(\d+)\.(.+)\.weight$/);
  if (match === null) {
    return null;
  }
  const [, index, local] = match;
  if (index === undefined || local === undefined) {
    return null;
  }
  const path = SINGLE_BLOCK_WEIGHT_PATHS.get(local);
  return path === undefined ? null : `singleTransformerBlocks.${index}.${path}.weight`;
}

/** Map a Diffusers FLUX.2 transformer tensor name onto the package parameter tree. */
export function flux2KleinTransformerWeightPath(checkpointName: string): string | null {
  const doublePath = doubleBlockWeightPath(checkpointName);
  if (doublePath !== null) {
    return doublePath;
  }
  const singlePath = singleBlockWeightPath(checkpointName);
  if (singlePath !== null) {
    return singlePath;
  }
  return TOP_LEVEL_WEIGHT_PATHS.get(checkpointName) ?? null;
}

/** Transform a Diffusers FLUX.2 transformer tensor into package-owned parameter layout. */
export function transformFlux2KleinTransformerWeight(
  _weightPath: string,
  tensor: MxArray,
): MxArray {
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
  options: Flux2KleinTransformerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadFlux2KleinTransformerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function assignWeightTensor(
  model: Flux2KleinTransformer2DModel,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformFlux2KleinTransformerWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignFlux2KleinTransformerWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: Flux2KleinTransformer2DModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = flux2KleinTransformerWeightPath(checkpointName);
  if (path === null || !expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into a FLUX.2 Klein transformer module. */
export async function loadFlux2KleinTransformerWeights(
  model: Flux2KleinTransformer2DModel,
  component: DiffusionSnapshotComponent,
  options: Flux2KleinTransformerWeightLoadOptions = {},
): Promise<Flux2KleinTransformerWeightLoadResult> {
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

/** Construct and load the FLUX.2 Klein transformer component from a snapshot manifest. */
export async function loadFlux2KleinTransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Flux2KleinTransformerWeightLoadOptions = {},
): Promise<Flux2KleinTransformer2DModel> {
  const configs = await loadFlux2KleinComponentConfigs(manifest);
  const model = new Flux2KleinTransformer2DModel(configs.transformer);
  try {
    await loadFlux2KleinTransformerWeights(
      model,
      flux2KleinTransformerComponent(manifest),
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
