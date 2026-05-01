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
import { loadLtxComponentConfigs } from "./config";
import { LtxVideoTransformer3DModel } from "./transformer";

export type LtxVideoTransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-Video transformer weights. */
export type LtxVideoTransformerWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  ["proj_in.weight", "projIn.weight"],
  ["proj_in.bias", "projIn.bias"],
  ["scale_shift_table", "scaleShiftTable"],
  ["time_embed.linear.weight", "timeEmbed.linear.weight"],
  ["time_embed.linear.bias", "timeEmbed.linear.bias"],
  [
    "time_embed.emb.timestep_embedder.linear_1.weight",
    "timeEmbed.emb.timestepEmbedder.linear1.weight",
  ],
  ["time_embed.emb.timestep_embedder.linear_1.bias", "timeEmbed.emb.timestepEmbedder.linear1.bias"],
  [
    "time_embed.emb.timestep_embedder.linear_2.weight",
    "timeEmbed.emb.timestepEmbedder.linear2.weight",
  ],
  ["time_embed.emb.timestep_embedder.linear_2.bias", "timeEmbed.emb.timestepEmbedder.linear2.bias"],
  ["caption_projection.linear_1.weight", "captionProjection.linear1.weight"],
  ["caption_projection.linear_1.bias", "captionProjection.linear1.bias"],
  ["caption_projection.linear_2.weight", "captionProjection.linear2.weight"],
  ["caption_projection.linear_2.bias", "captionProjection.linear2.bias"],
  ["proj_out.weight", "projOut.weight"],
  ["proj_out.bias", "projOut.bias"],
]);

const ATTENTION_WEIGHT_PATHS = new Map<string, string>([
  ["norm_q.weight", "normQ.weight"],
  ["norm_k.weight", "normK.weight"],
  ["to_q.weight", "toQ.weight"],
  ["to_q.bias", "toQ.bias"],
  ["to_k.weight", "toK.weight"],
  ["to_k.bias", "toK.bias"],
  ["to_v.weight", "toV.weight"],
  ["to_v.bias", "toV.bias"],
  ["to_out.0.weight", "toOut.weight"],
  ["to_out.0.bias", "toOut.bias"],
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
        `assignLtxVideoTransformerWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignLtxVideoTransformerWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignLtxVideoTransformerWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignLtxVideoTransformerWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignLtxVideoTransformerWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtxVideoTransformerWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignLtxVideoTransformerWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function ltxVideoTransformerComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "transformer" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError(
      "LTX-Video snapshot manifest is missing an enabled transformer.",
    );
  }
  return component;
}

function blockAttentionWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.(attn[12])\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, attentionName, local] = match;
  if (index === undefined || attentionName === undefined || local === undefined) {
    return null;
  }
  const mapped = ATTENTION_WEIGHT_PATHS.get(local);
  if (mapped === undefined) {
    return null;
  }
  return `transformerBlocks.${index}.${attentionName}.${mapped}`;
}

function blockFeedForwardWeightPath(checkpointName: string): string | null {
  const first = checkpointName.match(/^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.+)$/);
  if (first !== null) {
    const [, index, leaf] = first;
    return index === undefined || leaf === undefined
      ? null
      : `transformerBlocks.${index}.ff.linear1.${leaf}`;
  }
  const second = checkpointName.match(/^transformer_blocks\.(\d+)\.ff\.net\.2\.(.+)$/);
  if (second !== null) {
    const [, index, leaf] = second;
    return index === undefined || leaf === undefined
      ? null
      : `transformerBlocks.${index}.ff.linear2.${leaf}`;
  }
  return null;
}

function blockScaleShiftWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.scale_shift_table$/);
  if (match === null) {
    return null;
  }
  const [, index] = match;
  return index === undefined ? null : `transformerBlocks.${index}.scaleShiftTable`;
}

/** Map a Diffusers LTX-Video transformer tensor name onto the package parameter tree. */
export function ltxVideoTransformerWeightPath(checkpointName: string): string | null {
  return (
    blockAttentionWeightPath(checkpointName) ??
    blockFeedForwardWeightPath(checkpointName) ??
    blockScaleShiftWeightPath(checkpointName) ??
    TOP_LEVEL_WEIGHT_PATHS.get(checkpointName) ??
    null
  );
}

/** Transform a Diffusers LTX-Video transformer tensor into package-owned parameter layout. */
export function transformLtxVideoTransformerWeight(_weightPath: string, tensor: MxArray): MxArray {
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
  options: LtxVideoTransformerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtxVideoTransformerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function assignWeightTensor(
  model: LtxVideoTransformer3DModel,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtxVideoTransformerWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtxVideoTransformerWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: LtxVideoTransformer3DModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltxVideoTransformerWeightPath(checkpointName);
  if (path === null || !expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into an LTX-Video transformer module. */
export async function loadLtxVideoTransformerWeights(
  model: LtxVideoTransformer3DModel,
  component: DiffusionSnapshotComponent,
  options: LtxVideoTransformerWeightLoadOptions = {},
): Promise<LtxVideoTransformerWeightLoadResult> {
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

/** Construct and load the classic LTX-Video transformer component from a snapshot manifest. */
export async function loadLtxVideoTransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: LtxVideoTransformerWeightLoadOptions = {},
): Promise<LtxVideoTransformer3DModel> {
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx-video") {
    throw new DiffusionConfigError("loadLtxVideoTransformerFromSnapshot requires LTX-Video.");
  }
  const model = new LtxVideoTransformer3DModel(configs.transformer);
  try {
    await loadLtxVideoTransformerWeights(model, ltxVideoTransformerComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
