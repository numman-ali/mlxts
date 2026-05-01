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
import { Ltx2TextConnectors } from "./connectors-ltx2";

export type Ltx2TextConnectorWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading LTX-2 text connector weights. */
export type Ltx2TextConnectorWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

type SafetensorsIndexWeightMap = Record<string, string>;

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  ["text_proj_in.weight", "textProjIn.weight"],
  ["text_proj_in.bias", "textProjIn.bias"],
  ["video_text_proj_in.weight", "videoTextProjIn.weight"],
  ["video_text_proj_in.bias", "videoTextProjIn.bias"],
  ["audio_text_proj_in.weight", "audioTextProjIn.weight"],
  ["audio_text_proj_in.bias", "audioTextProjIn.bias"],
]);

const CONNECTOR_PREFIXES = new Map<string, string>([
  ["video_connector", "videoConnector"],
  ["audio_connector", "audioConnector"],
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
  ["to_gate_logits.weight", "toGateLogits.weight"],
  ["to_gate_logits.bias", "toGateLogits.bias"],
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
        `assignLtx2TextConnectorWeightPath: "${path}" segment "${segment}" is invalid.`,
      );
    }
    return current[index];
  }
  if (typeof current !== "object" || current === null) {
    throw new Error(
      `assignLtx2TextConnectorWeightPath: "${path}" cannot descend through a non-object.`,
    );
  }
  return Reflect.get(current, segment);
}

function assignLtx2TextConnectorWeightPath(root: object, path: string, tensor: MxArray): void {
  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(
        `assignLtx2TextConnectorWeightPath: "${path}" contains an undefined segment.`,
      );
    }
    current = nextNode(current, segment, path);
  }
  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(
      `assignLtx2TextConnectorWeightPath: "${path}" does not point to an object property.`,
    );
  }
  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignLtx2TextConnectorWeightPath: "${path}" is missing a leaf segment.`);
  }
  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(
      `assignLtx2TextConnectorWeightPath: "${path}" does not point to an MxArray parameter.`,
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

function ltx2TextConnectorsComponent(
  manifest: DiffusionSnapshotManifest,
): DiffusionSnapshotComponent {
  const component = manifest.components.find(
    (candidate) => candidate.name === "connectors" && candidate.enabled,
  );
  if (component === undefined) {
    throw new DiffusionConfigError("LTX-2 snapshot manifest is missing enabled connectors.");
  }
  return component;
}

function connectorPrefix(checkpointName: string): { source: string; target: string } | null {
  const match = checkpointName.match(/^(video_connector|audio_connector)\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, rawPrefix, source] = match;
  if (rawPrefix === undefined || source === undefined) {
    return null;
  }
  const target = CONNECTOR_PREFIXES.get(rawPrefix);
  return target === undefined ? null : { source, target };
}

function connectorRegisterWeightPath(checkpointName: string): string | null {
  const prefixed = connectorPrefix(checkpointName);
  if (prefixed === null || prefixed.source !== "learnable_registers") {
    return null;
  }
  return `${prefixed.target}.learnableRegisters`;
}

function connectorAttentionWeightPath(checkpointName: string): string | null {
  const prefixed = connectorPrefix(checkpointName);
  if (prefixed === null) {
    return null;
  }
  const match = prefixed.source.match(/^transformer_blocks\.(\d+)\.attn1\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, local] = match;
  if (index === undefined || local === undefined) {
    return null;
  }
  const mapped = ATTENTION_WEIGHT_PATHS.get(local);
  return mapped === undefined
    ? null
    : `${prefixed.target}.transformerBlocks.${index}.attn1.${mapped}`;
}

function connectorFeedForwardWeightPath(checkpointName: string): string | null {
  const prefixed = connectorPrefix(checkpointName);
  if (prefixed === null) {
    return null;
  }
  const match = prefixed.source.match(/^transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, layer, leaf] = match;
  if (index === undefined || layer === undefined || leaf === undefined) {
    return null;
  }
  const mappedLayer = layer === "0.proj" ? "linear1" : "linear2";
  return `${prefixed.target}.transformerBlocks.${index}.ff.${mappedLayer}.${leaf}`;
}

/** Map a Diffusers LTX-2 text connector tensor name onto the package parameter tree. */
export function ltx2TextConnectorWeightPath(checkpointName: string): string | null {
  return (
    connectorRegisterWeightPath(checkpointName) ??
    connectorAttentionWeightPath(checkpointName) ??
    connectorFeedForwardWeightPath(checkpointName) ??
    TOP_LEVEL_WEIGHT_PATHS.get(checkpointName) ??
    null
  );
}

/** Transform a Diffusers LTX-2 connector tensor into package-owned parameter layout. */
export function transformLtx2TextConnectorWeight(_weightPath: string, tensor: MxArray): MxArray {
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
  options: Ltx2TextConnectorWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadLtx2TextConnectorWeights: checkpoint contained unexpected unmapped weights: ${[
      ...unexpectedWeights,
    ]
      .toSorted((left, right) => left.localeCompare(right))
      .join(", ")}.`,
  );
}

function assignWeightTensor(
  model: Ltx2TextConnectors,
  path: string,
  tensor: MxArray,
  assignedPaths: Set<string>,
): void {
  let assignedTensor: MxArray | null = tensor;
  try {
    const transformed = transformLtx2TextConnectorWeight(path, tensor);
    if (transformed !== tensor) {
      tensor.free();
    }
    assignedTensor = transformed;
    assignLtx2TextConnectorWeightPath(model, path, assignedTensor);
    assignedPaths.add(path);
    assignedTensor = null;
  } finally {
    assignedTensor?.free();
  }
}

function consumeCheckpointTensor(
  model: Ltx2TextConnectors,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: Set<string>,
  unexpectedWeights: string[],
  checkpointName: string,
  tensor: MxArray,
): void {
  const path = ltx2TextConnectorWeightPath(checkpointName);
  if (path === null || !expectedPaths.has(path)) {
    unexpectedWeights.push(checkpointName);
    tensor.free();
    return;
  }
  assignWeightTensor(model, path, tensor, assignedPaths);
}

/** Load Diffusers safetensors weights into an LTX-2 text connector module. */
export async function loadLtx2TextConnectorWeights(
  model: Ltx2TextConnectors,
  component: DiffusionSnapshotComponent,
  options: Ltx2TextConnectorWeightLoadOptions = {},
): Promise<Ltx2TextConnectorWeightLoadResult> {
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

/** Construct and load the LTX-2 text connector component from a snapshot manifest. */
export async function loadLtx2TextConnectorsFromSnapshot(
  manifest: DiffusionSnapshotManifest,
  options: Ltx2TextConnectorWeightLoadOptions = {},
): Promise<Ltx2TextConnectors> {
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx2") {
    throw new DiffusionConfigError("loadLtx2TextConnectorsFromSnapshot requires LTX2Pipeline.");
  }
  const model = new Ltx2TextConnectors(configs.connectors);
  try {
    await loadLtx2TextConnectorWeights(model, ltx2TextConnectorsComponent(manifest), options);
    model.eval();
    const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
    mxEval(...parameters);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
