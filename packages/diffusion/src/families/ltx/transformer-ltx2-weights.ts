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
import { Ltx2VideoTransformer3DModel } from "./transformer-ltx2";

export type Ltx2VideoTransformerWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-2 transformer weights. */
export type Ltx2VideoTransformerWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  ["proj_in.weight", "projIn.weight"],
  ["proj_in.bias", "projIn.bias"],
  ["audio_proj_in.weight", "audioProjIn.weight"],
  ["audio_proj_in.bias", "audioProjIn.bias"],
  ["scale_shift_table", "scaleShiftTable"],
  ["audio_scale_shift_table", "audioScaleShiftTable"],
  ["proj_out.weight", "projOut.weight"],
  ["proj_out.bias", "projOut.bias"],
  ["audio_proj_out.weight", "audioProjOut.weight"],
  ["audio_proj_out.bias", "audioProjOut.bias"],
  ["caption_projection.linear_1.weight", "captionProjection.linear1.weight"],
  ["caption_projection.linear_1.bias", "captionProjection.linear1.bias"],
  ["caption_projection.linear_2.weight", "captionProjection.linear2.weight"],
  ["caption_projection.linear_2.bias", "captionProjection.linear2.bias"],
  ["audio_caption_projection.linear_1.weight", "audioCaptionProjection.linear1.weight"],
  ["audio_caption_projection.linear_1.bias", "audioCaptionProjection.linear1.bias"],
  ["audio_caption_projection.linear_2.weight", "audioCaptionProjection.linear2.weight"],
  ["audio_caption_projection.linear_2.bias", "audioCaptionProjection.linear2.bias"],
]);

const EMBEDDER_WEIGHT_PREFIXES = new Map<string, string>([
  ["time_embed", "timeEmbed"],
  ["audio_time_embed", "audioTimeEmbed"],
  ["av_cross_attn_video_scale_shift", "avCrossAttnVideoScaleShift"],
  ["av_cross_attn_audio_scale_shift", "avCrossAttnAudioScaleShift"],
  ["av_cross_attn_video_a2v_gate", "avCrossAttnVideoA2vGate"],
  ["av_cross_attn_audio_v2a_gate", "avCrossAttnAudioV2aGate"],
]);

const ATTENTION_NAMES = new Map<string, string>([
  ["attn1", "attn1"],
  ["audio_attn1", "audioAttn1"],
  ["attn2", "attn2"],
  ["audio_attn2", "audioAttn2"],
  ["audio_to_video_attn", "audioToVideoAttn"],
  ["video_to_audio_attn", "videoToAudioAttn"],
]);

const ATTENTION_WEIGHT_PATHS = new Map<string, string>([
  ["to_q.weight", "toQ.weight"],
  ["to_q.bias", "toQ.bias"],
  ["to_k.weight", "toK.weight"],
  ["to_k.bias", "toK.bias"],
  ["to_v.weight", "toV.weight"],
  ["to_v.bias", "toV.bias"],
  ["to_out.0.weight", "toOut.weight"],
  ["to_out.0.bias", "toOut.bias"],
]);

const BLOCK_TABLE_PATHS = new Map<string, string>([
  ["scale_shift_table", "scaleShiftTable"],
  ["audio_scale_shift_table", "audioScaleShiftTable"],
  ["video_a2v_cross_attn_scale_shift_table", "videoA2vCrossAttnScaleShiftTable"],
  ["audio_a2v_cross_attn_scale_shift_table", "audioA2vCrossAttnScaleShiftTable"],
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
        `assignLtx2VideoTransformerWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignLtx2VideoTransformerWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignLtx2VideoTransformerWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignLtx2VideoTransformerWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignLtx2VideoTransformerWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtx2VideoTransformerWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignLtx2VideoTransformerWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function ltx2VideoTransformerComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "transformer" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("LTX-2 snapshot manifest is missing an enabled transformer.");
  }
  return component;
}

function embedderWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^(.+)\.(linear|emb\.timestep_embedder\.linear_[12])\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, prefix, local, leaf] = match;
  if (prefix === undefined || local === undefined || leaf === undefined) {
    return null;
  }
  const mappedPrefix = EMBEDDER_WEIGHT_PREFIXES.get(prefix);
  if (mappedPrefix === undefined || (leaf !== "weight" && leaf !== "bias")) {
    return null;
  }
  if (local === "linear") {
    return `${mappedPrefix}.linear.${leaf}`;
  }
  const layer = local.endsWith("linear_1") ? "linear1" : "linear2";
  return `${mappedPrefix}.emb.timestepEmbedder.${layer}.${leaf}`;
}

function blockAttentionWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.([^.]+)\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, attentionName, local] = match;
  if (index === undefined || attentionName === undefined || local === undefined) {
    return null;
  }
  const mappedAttentionName = ATTENTION_NAMES.get(attentionName);
  const mapped = ATTENTION_WEIGHT_PATHS.get(local);
  if (mappedAttentionName === undefined || mapped === undefined) {
    return null;
  }
  return `transformerBlocks.${index}.${mappedAttentionName}.${mapped}`;
}

function blockFeedForwardWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(
    /^transformer_blocks\.(\d+)\.(ff|audio_ff)\.net\.(0\.proj|2)\.(.+)$/,
  );
  if (match === null) {
    return null;
  }
  const [, index, name, layer, leaf] = match;
  if (index === undefined || name === undefined || layer === undefined || leaf === undefined) {
    return null;
  }
  const mappedName = name === "ff" ? "ff" : "audioFf";
  const mappedLayer = layer === "0.proj" ? "linear1" : "linear2";
  return `transformerBlocks.${index}.${mappedName}.${mappedLayer}.${leaf}`;
}

function blockTableWeightPath(checkpointName: string): string | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.([^.]+)$/);
  if (match === null) {
    return null;
  }
  const [, index, local] = match;
  if (index === undefined || local === undefined) {
    return null;
  }
  const mapped = BLOCK_TABLE_PATHS.get(local);
  return mapped === undefined ? null : `transformerBlocks.${index}.${mapped}`;
}

/** Map a Diffusers LTX-2 transformer tensor name onto the package parameter tree. */
export function ltx2VideoTransformerWeightPath(checkpointName: string): string | null {
  return (
    blockAttentionWeightPath(checkpointName) ??
    blockFeedForwardWeightPath(checkpointName) ??
    blockTableWeightPath(checkpointName) ??
    embedderWeightPath(checkpointName) ??
    TOP_LEVEL_WEIGHT_PATHS.get(checkpointName) ??
    null
  );
}

/** Transform a Diffusers LTX-2 transformer tensor into package-owned parameter layout. */
export function transformLtx2VideoTransformerWeight(_weightPath: string, tensor: MxArray): MxArray {
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
  options: Ltx2VideoTransformerWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtx2VideoTransformerWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function assignWeightTensor(
  model: Ltx2VideoTransformer3DModel,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtx2VideoTransformerWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtx2VideoTransformerWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: Ltx2VideoTransformer3DModel,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltx2VideoTransformerWeightPath(checkpointName);
  if (path === null || !expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into an LTX-2 transformer module. */
export async function loadLtx2VideoTransformerWeights(
  model: Ltx2VideoTransformer3DModel,
  component: DiffusionSnapshotComponent,
  options: Ltx2VideoTransformerWeightLoadOptions = {},
): Promise<Ltx2VideoTransformerWeightLoadResult> {
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

/** Construct and load the LTX-2 transformer component from a snapshot manifest. */
export async function loadLtx2VideoTransformerFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Ltx2VideoTransformerWeightLoadOptions = {},
): Promise<Ltx2VideoTransformer3DModel> {
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx2") {
    throw new DiffusionConfigError("loadLtx2VideoTransformerFromSnapshot requires LTX2Pipeline.");
  }
  const model = new Ltx2VideoTransformer3DModel(configs.transformer);
  try {
    await loadLtx2VideoTransformerWeights(model, ltx2VideoTransformerComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
