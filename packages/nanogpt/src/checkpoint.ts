/**
 * Canonical checkpointing for nanogpt.
 *
 * Checkpoints are directory-based and forward-only:
 * - manifest.json for metadata
 * - tensors.bin for raw tensor bytes
 *
 * The format is intentionally simple and explicit. This repo only carries
 * the canonical checkpoint format.
 *
 * @module
 */

import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "fs";
import { AdamW, type DType, MxArray, mxEval, treeFlatten, treeUnflatten } from "mlx-ts";
import { basename, dirname, join } from "path";
import type { GPTConfig } from "./config";
import type { GPT } from "./model/gpt";
import type { CharTokenizer } from "./tokenizer";

const CHECKPOINT_VERSION = 2;
const MANIFEST_FILENAME = "manifest.json";
const TENSOR_DATA_FILENAME = "tensors.bin";
export type CheckpointKind = "snapshot" | "resume";

type SupportedCheckpointDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";

type CheckpointTensorMeta = {
  shape: number[];
  dtype: SupportedCheckpointDType;
  offset: number;
  byteLength: number;
};

type CheckpointManifest = {
  version: 2;
  kind: CheckpointKind;
  config: GPTConfig;
  step: number;
  tokenizer: {
    chars: string[];
  };
  parameters: Record<string, CheckpointTensorMeta>;
  optimizer?: AdamWOptimizerManifest;
};

type AdamWOptimizerManifest = {
  kind: "adamw";
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  state: Record<string, Record<string, CheckpointTensorMeta>>;
};

/** Serialized checkpoint tensor payload. */
export interface CheckpointTensor {
  shape: number[];
  dtype: SupportedCheckpointDType;
  data: Uint8Array;
}

/** In-memory checkpoint representation. */
export interface CheckpointData {
  version: 2;
  kind: CheckpointKind;
  config: GPTConfig;
  step: number;
  tokenizer: {
    chars: string[];
  };
  parameters: Record<string, CheckpointTensor>;
  optimizer?: AdamWOptimizerCheckpoint;
}

/** Serialized AdamW optimizer checkpoint payload. */
export interface AdamWOptimizerCheckpoint {
  kind: "adamw";
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  state: Record<string, Record<string, CheckpointTensor>>;
}

type SaveCheckpointOptions = {
  model: GPT;
  kind: CheckpointKind;
  config: GPTConfig;
  step: number;
  tokenizer: CharTokenizer;
  path: string;
  optimizer?: AdamW | undefined;
};

type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context}: expected a finite number`);
  }
  return value;
}

function readString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context}: expected a string`);
  }
  return value;
}

function readBoolean(value: unknown, context: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${context}: expected a boolean`);
  }
  return value;
}

function readStringArray(value: unknown, context: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a string array`);
  }

  const chars: string[] = [];
  for (let index = 0; index < value.length; index++) {
    chars.push(readString(value[index], `${context}[${index}]`));
  }
  return chars;
}

function readShape(value: unknown, context: string): number[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a shape array`);
  }

  const shape: number[] = [];
  for (let index = 0; index < value.length; index++) {
    const dimension = readNumber(value[index], `${context}[${index}]`);
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(`${context}[${index}]: expected a non-negative integer`);
    }
    shape.push(dimension);
  }
  return shape;
}

function isSupportedCheckpointDType(value: string): value is SupportedCheckpointDType {
  return (
    value === "bool" ||
    value === "uint8" ||
    value === "uint16" ||
    value === "uint32" ||
    value === "int8" ||
    value === "int16" ||
    value === "int32" ||
    value === "float32" ||
    value === "float64"
  );
}

function readDType(value: unknown, context: string): SupportedCheckpointDType {
  const dtype = readString(value, context);
  if (!isSupportedCheckpointDType(dtype)) {
    throw new Error(`${context}: unsupported checkpoint dtype "${dtype}"`);
  }
  return dtype;
}

function readTokenizer(value: unknown): { chars: string[] } {
  if (!isRecord(value)) {
    throw new Error("checkpoint manifest tokenizer: expected an object");
  }
  return { chars: readStringArray(value.chars, "checkpoint manifest tokenizer.chars") };
}

function readConfig(value: unknown): GPTConfig {
  if (!isRecord(value)) {
    throw new Error("checkpoint manifest config: expected an object");
  }

  const config: GPTConfig = {
    nLayer: readNumber(value.nLayer, "checkpoint manifest config.nLayer"),
    nHead: readNumber(value.nHead, "checkpoint manifest config.nHead"),
    nEmbd: readNumber(value.nEmbd, "checkpoint manifest config.nEmbd"),
    blockSize: readNumber(value.blockSize, "checkpoint manifest config.blockSize"),
    dropout: readNumber(value.dropout, "checkpoint manifest config.dropout"),
    gradientCheckpointing: readBoolean(
      value.gradientCheckpointing,
      "checkpoint manifest config.gradientCheckpointing",
    ),
    vocabSize: readNumber(value.vocabSize, "checkpoint manifest config.vocabSize"),
  };

  if (!Number.isInteger(config.nLayer) || config.nLayer <= 0) {
    throw new Error(`checkpoint manifest config.nLayer: expected a positive integer`);
  }
  if (!Number.isInteger(config.nHead) || config.nHead <= 0) {
    throw new Error(`checkpoint manifest config.nHead: expected a positive integer`);
  }
  if (!Number.isInteger(config.nEmbd) || config.nEmbd <= 0) {
    throw new Error(`checkpoint manifest config.nEmbd: expected a positive integer`);
  }
  if (!Number.isInteger(config.blockSize) || config.blockSize <= 0) {
    throw new Error(`checkpoint manifest config.blockSize: expected a positive integer`);
  }
  if (config.dropout < 0 || config.dropout >= 1) {
    throw new Error(`checkpoint manifest config.dropout: expected a value in [0, 1)`);
  }
  if (!Number.isInteger(config.vocabSize) || config.vocabSize <= 0) {
    throw new Error(`checkpoint manifest config.vocabSize: expected a positive integer`);
  }
  if (config.nEmbd % config.nHead !== 0) {
    throw new Error(`checkpoint manifest config: nEmbd must be divisible by nHead`);
  }

  return config;
}

function bytesPerElement(dtype: SupportedCheckpointDType): number {
  switch (dtype) {
    case "bool":
    case "uint8":
    case "int8":
      return 1;
    case "uint16":
    case "int16":
      return 2;
    case "uint32":
    case "int32":
    case "float32":
      return 4;
    case "float64":
      return 8;
  }
}

function elementCount(shape: readonly number[]): number {
  return shape.length === 0 ? 1 : shape.reduce((product, dimension) => product * dimension, 1);
}

function readParameterMetadata(value: unknown, path: string): CheckpointTensorMeta {
  if (!isRecord(value)) {
    throw new Error(`checkpoint manifest parameter "${path}": expected an object`);
  }

  const shape = readShape(value.shape, `checkpoint manifest parameter "${path}".shape`);
  const dtype = readDType(value.dtype, `checkpoint manifest parameter "${path}".dtype`);
  const offset = readNumber(value.offset, `checkpoint manifest parameter "${path}".offset`);
  const byteLength = readNumber(
    value.byteLength,
    `checkpoint manifest parameter "${path}".byteLength`,
  );

  if (!Number.isInteger(offset) || offset < 0) {
    throw new Error(
      `checkpoint manifest parameter "${path}".offset: expected a non-negative integer`,
    );
  }
  if (!Number.isInteger(byteLength) || byteLength < 0) {
    throw new Error(
      `checkpoint manifest parameter "${path}".byteLength: expected a non-negative integer`,
    );
  }

  const expectedBytes = elementCount(shape) * bytesPerElement(dtype);
  if (byteLength !== expectedBytes) {
    throw new Error(
      `checkpoint manifest parameter "${path}": byteLength ${byteLength} does not match shape/dtype payload size ${expectedBytes}`,
    );
  }

  return { shape, dtype, offset, byteLength };
}

function readParameters(value: unknown): Record<string, CheckpointTensorMeta> {
  if (!isRecord(value)) {
    throw new Error("checkpoint manifest parameters: expected an object");
  }

  const parameters: Record<string, CheckpointTensorMeta> = {};
  for (const [path, entry] of Object.entries(value)) {
    parameters[path] = readParameterMetadata(entry, path);
  }
  return parameters;
}

function readOptimizerState(
  value: unknown,
  context: string,
): Record<string, Record<string, CheckpointTensorMeta>> {
  if (!isRecord(value)) {
    throw new Error(`${context}: expected an object`);
  }

  const state: Record<string, Record<string, CheckpointTensorMeta>> = {};
  for (const [path, slots] of Object.entries(value)) {
    if (!isRecord(slots)) {
      throw new Error(`${context}.${path}: expected an object`);
    }

    const slotState: Record<string, CheckpointTensorMeta> = {};
    for (const [slotName, slotValue] of Object.entries(slots)) {
      slotState[slotName] = readParameterMetadata(slotValue, `${path}.${slotName}`);
    }
    state[path] = slotState;
  }

  return state;
}

function readOptimizer(value: unknown): AdamWOptimizerManifest {
  if (!isRecord(value)) {
    throw new Error("checkpoint manifest optimizer: expected an object");
  }
  if (value.kind !== "adamw") {
    throw new Error(`checkpoint manifest optimizer.kind: expected "adamw"`);
  }

  const step = readNumber(value.step, "checkpoint manifest optimizer.step");
  if (!Number.isInteger(step) || step < 0) {
    throw new Error("checkpoint manifest optimizer.step: expected a non-negative integer");
  }

  return {
    kind: "adamw",
    step,
    lr: readNumber(value.lr, "checkpoint manifest optimizer.lr"),
    beta1: readNumber(value.beta1, "checkpoint manifest optimizer.beta1"),
    beta2: readNumber(value.beta2, "checkpoint manifest optimizer.beta2"),
    eps: readNumber(value.eps, "checkpoint manifest optimizer.eps"),
    weightDecay: readNumber(value.weightDecay, "checkpoint manifest optimizer.weightDecay"),
    state: readOptimizerState(value.state, "checkpoint manifest optimizer.state"),
  };
}

function readManifest(path: string): CheckpointManifest {
  const manifestPath = join(path, MANIFEST_FILENAME);
  const parsed: unknown = JSON.parse(readFileSync(manifestPath, "utf-8"));
  if (!isRecord(parsed)) {
    throw new Error("checkpoint manifest: expected an object");
  }

  const version = readNumber(parsed.version, "checkpoint manifest version");
  if (version !== CHECKPOINT_VERSION) {
    throw new Error(
      `checkpoint manifest version ${version} is unsupported; expected ${CHECKPOINT_VERSION}`,
    );
  }

  const kind = readString(parsed.kind, "checkpoint manifest kind");
  if (kind !== "snapshot" && kind !== "resume") {
    throw new Error(`checkpoint manifest kind "${kind}" is unsupported`);
  }

  const manifest: CheckpointManifest = {
    version: CHECKPOINT_VERSION,
    kind,
    config: readConfig(parsed.config),
    step: readNumber(parsed.step, "checkpoint manifest step"),
    tokenizer: readTokenizer(parsed.tokenizer),
    parameters: readParameters(parsed.parameters),
  };

  if (parsed.optimizer !== undefined) {
    manifest.optimizer = readOptimizer(parsed.optimizer);
  }
  if (manifest.kind === "resume" && manifest.optimizer === undefined) {
    throw new Error("checkpoint manifest: resume checkpoints require optimizer metadata");
  }
  if (manifest.kind === "snapshot" && manifest.optimizer !== undefined) {
    throw new Error(
      "checkpoint manifest: snapshot checkpoints must not include optimizer metadata",
    );
  }
  if (manifest.optimizer !== undefined && manifest.optimizer.step !== manifest.step) {
    throw new Error(
      `checkpoint manifest optimizer.step ${manifest.optimizer.step} does not match checkpoint step ${manifest.step}`,
    );
  }

  return manifest;
}

function checkpointDType(dtype: DType): SupportedCheckpointDType {
  if (!isSupportedCheckpointDType(dtype)) {
    throw new Error(`checkpoint: dtype "${dtype}" is not supported by the checkpoint format`);
  }
  return dtype;
}

function typedArrayToBytes(data: SupportedTypedArray): Uint8Array {
  const copied = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new Uint8Array(copied);
}

function typedArrayFromBytes(
  bytes: Uint8Array,
  dtype: SupportedCheckpointDType,
): SupportedTypedArray {
  const copied = Uint8Array.from(bytes);
  const buffer = copied.buffer;
  switch (dtype) {
    case "bool":
    case "uint8":
      return new Uint8Array(buffer);
    case "uint16":
      return new Uint16Array(buffer);
    case "uint32":
      return new Uint32Array(buffer);
    case "int8":
      return new Int8Array(buffer);
    case "int16":
      return new Int16Array(buffer);
    case "int32":
      return new Int32Array(buffer);
    case "float32":
      return new Float32Array(buffer);
    case "float64":
      return new Float64Array(buffer);
  }
}

function sortedParameterEntries(model: GPT): [string[], MxArray][] {
  return [...treeFlatten(model.parameters())].sort(([left], [right]) =>
    left.join(".").localeCompare(right.join(".")),
  );
}

function serializeTensorEntries(entries: Array<{ key: string; value: MxArray }>): {
  bytes: Uint8Array;
  parameters: Record<string, CheckpointTensorMeta>;
} {
  const arrays = entries.map(({ value }) => value);
  mxEval(...arrays);

  const serialized = entries.map(({ key, value }) => {
    const dtype = checkpointDType(value.dtype);
    const data = typedArrayToBytes(value.toTypedArray());
    return {
      key,
      shape: [...value.shape],
      dtype,
      data,
    };
  });

  let totalBytes = 0;
  for (const entry of serialized) {
    totalBytes += entry.data.byteLength;
  }

  const bytes = new Uint8Array(totalBytes);
  const parameters: Record<string, CheckpointTensorMeta> = {};
  let offset = 0;

  for (const entry of serialized) {
    bytes.set(entry.data, offset);
    parameters[entry.key] = {
      shape: entry.shape,
      dtype: entry.dtype,
      offset,
      byteLength: entry.data.byteLength,
    };
    offset += entry.data.byteLength;
  }

  return { bytes, parameters };
}

function serializeParameters(model: GPT): {
  bytes: Uint8Array;
  parameters: Record<string, CheckpointTensorMeta>;
} {
  return serializeTensorEntries(
    sortedParameterEntries(model).map(([path, value]) => ({ key: path.join("."), value })),
  );
}

function serializeOptimizer(optimizer: AdamW): {
  bytes: Uint8Array;
  optimizer: AdamWOptimizerManifest;
} {
  const checkpoint = optimizer.checkpoint();
  const tensorEntries: Array<{ key: string; value: MxArray }> = [];

  for (const [path, slots] of Object.entries(checkpoint.state)) {
    for (const [slotName, value] of Object.entries(slots)) {
      tensorEntries.push({ key: `${path}::${slotName}`, value });
    }
  }

  const { bytes, parameters } = serializeTensorEntries(tensorEntries);
  const state: Record<string, Record<string, CheckpointTensorMeta>> = {};

  for (const [path, slots] of Object.entries(checkpoint.state)) {
    const slotState: Record<string, CheckpointTensorMeta> = {};
    for (const slotName of Object.keys(slots)) {
      const entry = parameters[`${path}::${slotName}`];
      if (entry === undefined) {
        throw new Error(
          `checkpoint optimizer serialization: missing state entry ${path}::${slotName}`,
        );
      }
      slotState[slotName] = entry;
    }
    state[path] = slotState;
  }

  return {
    bytes,
    optimizer: {
      kind: "adamw",
      step: checkpoint.step,
      lr: checkpoint.lr,
      beta1: checkpoint.beta1,
      beta2: checkpoint.beta2,
      eps: checkpoint.eps,
      weightDecay: checkpoint.weightDecay,
      state,
    },
  };
}

function shiftTensorMeta(meta: CheckpointTensorMeta, baseOffset: number): CheckpointTensorMeta {
  return {
    ...meta,
    offset: meta.offset + baseOffset,
  };
}

function shiftOptimizerOffsets(
  state: Record<string, Record<string, CheckpointTensorMeta>>,
  baseOffset: number,
): Record<string, Record<string, CheckpointTensorMeta>> {
  const shifted: Record<string, Record<string, CheckpointTensorMeta>> = {};
  for (const [path, slots] of Object.entries(state)) {
    const slotState: Record<string, CheckpointTensorMeta> = {};
    for (const [slotName, meta] of Object.entries(slots)) {
      slotState[slotName] = shiftTensorMeta(meta, baseOffset);
    }
    shifted[path] = slotState;
  }
  return shifted;
}

function readTensorSlice(
  key: string,
  meta: CheckpointTensorMeta,
  tensorBytes: Uint8Array,
): CheckpointTensor {
  const end = meta.offset + meta.byteLength;
  if (end > tensorBytes.byteLength) {
    throw new Error(
      `checkpoint tensor "${key}" exceeds tensors.bin size (${end} > ${tensorBytes.byteLength})`,
    );
  }

  return {
    shape: meta.shape,
    dtype: meta.dtype,
    data: tensorBytes.subarray(meta.offset, end),
  };
}

function loadOptimizerState(
  state: Record<string, Record<string, CheckpointTensorMeta>>,
  tensorBytes: Uint8Array,
): Record<string, Record<string, CheckpointTensor>> {
  const loaded: Record<string, Record<string, CheckpointTensor>> = {};
  for (const [path, slots] of Object.entries(state)) {
    const slotState: Record<string, CheckpointTensor> = {};
    for (const [slotName, meta] of Object.entries(slots)) {
      slotState[slotName] = readTensorSlice(`${path}.${slotName}`, meta, tensorBytes);
    }
    loaded[path] = slotState;
  }
  return loaded;
}

function checkpointTempPath(path: string): string {
  return join(
    dirname(path),
    `.${basename(path) || "checkpoint"}.${process.pid}.${crypto.randomUUID()}.tmp`,
  );
}

function checkpointBackupPath(path: string): string {
  return join(
    dirname(path),
    `.${basename(path) || "checkpoint"}.${process.pid}.${crypto.randomUUID()}.bak`,
  );
}

function writeCheckpointDirectory(
  path: string,
  manifest: CheckpointManifest,
  bytes: Uint8Array,
): void {
  const tempPath = checkpointTempPath(path);
  const backupPath = checkpointBackupPath(path);
  let backupCreated = false;
  let renamedIntoPlace = false;

  rmSync(tempPath, { recursive: true, force: true });
  mkdirSync(tempPath, { recursive: true });

  try {
    writeFileSync(
      join(tempPath, MANIFEST_FILENAME),
      `${JSON.stringify(manifest, null, 2)}\n`,
      "utf-8",
    );
    writeFileSync(join(tempPath, TENSOR_DATA_FILENAME), bytes);

    if (existsSync(path)) {
      rmSync(backupPath, { recursive: true, force: true });
      renameSync(path, backupPath);
      backupCreated = true;
    }

    renameSync(tempPath, path);
    renamedIntoPlace = true;
  } catch (error) {
    if (backupCreated && !renamedIntoPlace && !existsSync(path)) {
      renameSync(backupPath, path);
      backupCreated = false;
    }
    rmSync(tempPath, { recursive: true, force: true });
    throw error;
  } finally {
    if (backupCreated) {
      rmSync(backupPath, { recursive: true, force: true });
    }
  }
}

/**
 * Save a model checkpoint to a directory.
 *
 * The directory is staged to a sibling temp path first, then swapped into
 * place so a previously valid checkpoint is kept until the new one is ready.
 */
export function saveCheckpoint(options: SaveCheckpointOptions): void {
  const { model, optimizer, kind, config, step, tokenizer, path } = options;
  if (kind === "resume" && optimizer === undefined) {
    throw new Error("saveCheckpoint: resume checkpoints require optimizer state");
  }
  if (kind === "snapshot" && optimizer !== undefined) {
    throw new Error("saveCheckpoint: snapshot checkpoints must not include optimizer state");
  }
  if (optimizer !== undefined && optimizer.step !== step) {
    throw new Error(
      `saveCheckpoint: optimizer step ${optimizer.step} does not match checkpoint step ${step}`,
    );
  }
  const modelData = serializeParameters(model);
  const optimizerData = optimizer === undefined ? undefined : serializeOptimizer(optimizer);

  const totalBytes = modelData.bytes.byteLength + (optimizerData?.bytes.byteLength ?? 0);
  const bytes = new Uint8Array(totalBytes);
  bytes.set(modelData.bytes, 0);
  if (optimizerData !== undefined) {
    bytes.set(optimizerData.bytes, modelData.bytes.byteLength);
  }

  const manifest: CheckpointManifest = {
    version: CHECKPOINT_VERSION,
    kind,
    config,
    step,
    tokenizer: { chars: tokenizer.vocab },
    parameters: modelData.parameters,
  };
  if (optimizerData !== undefined) {
    manifest.optimizer = {
      ...optimizerData.optimizer,
      state: shiftOptimizerOffsets(optimizerData.optimizer.state, modelData.bytes.byteLength),
    };
  }

  writeCheckpointDirectory(path, manifest, bytes);
}

/**
 * Load a checkpoint directory into memory.
 *
 * This format is forward-only and canonical.
 */
export function loadCheckpoint(path: string): CheckpointData {
  const manifest = readManifest(path);
  const tensorBytes = new Uint8Array(readFileSync(join(path, TENSOR_DATA_FILENAME)));
  const parameters: Record<string, CheckpointTensor> = {};

  for (const [key, meta] of Object.entries(manifest.parameters)) {
    parameters[key] = readTensorSlice(key, meta, tensorBytes);
  }

  const checkpoint: CheckpointData = {
    version: CHECKPOINT_VERSION,
    kind: manifest.kind,
    config: manifest.config,
    step: manifest.step,
    tokenizer: manifest.tokenizer,
    parameters,
  };
  if (manifest.optimizer !== undefined) {
    checkpoint.optimizer = {
      kind: "adamw",
      step: manifest.optimizer.step,
      lr: manifest.optimizer.lr,
      beta1: manifest.optimizer.beta1,
      beta2: manifest.optimizer.beta2,
      eps: manifest.optimizer.eps,
      weightDecay: manifest.optimizer.weightDecay,
      state: loadOptimizerState(manifest.optimizer.state, tensorBytes),
    };
  }
  return checkpoint;
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function currentParameterMap(model: GPT): Map<string, { path: string[]; value: MxArray }> {
  const map = new Map<string, { path: string[]; value: MxArray }>();
  for (const [path, value] of sortedParameterEntries(model)) {
    map.set(path.join("."), { path, value });
  }
  return map;
}

function assertCheckpointMatchesModel(
  modelParameters: ReadonlyMap<string, { path: string[]; value: MxArray }>,
  checkpointParameters: Record<string, CheckpointTensor>,
): void {
  const checkpointKeys = new Set(Object.keys(checkpointParameters));

  for (const key of modelParameters.keys()) {
    const checkpointValue = checkpointParameters[key];
    if (checkpointValue === undefined) {
      throw new Error(`applyCheckpoint: missing checkpoint parameter "${key}"`);
    }

    const modelValue = modelParameters.get(key);
    if (modelValue === undefined) {
      throw new Error(`applyCheckpoint: model parameter "${key}" was unexpectedly unavailable`);
    }

    if (modelValue.value.dtype !== checkpointValue.dtype) {
      throw new Error(
        `applyCheckpoint: dtype mismatch for "${key}" (${checkpointValue.dtype} checkpoint vs ${modelValue.value.dtype} model)`,
      );
    }
    if (!sameShape(modelValue.value.shape, checkpointValue.shape)) {
      throw new Error(
        `applyCheckpoint: shape mismatch for "${key}" ([${checkpointValue.shape}] checkpoint vs [${modelValue.value.shape}] model)`,
      );
    }
  }

  for (const key of checkpointKeys) {
    if (!modelParameters.has(key)) {
      throw new Error(`applyCheckpoint: unexpected checkpoint parameter "${key}"`);
    }
  }
}

function createCheckpointArray(tensor: CheckpointTensor): MxArray {
  return MxArray.fromData(
    typedArrayFromBytes(tensor.data, tensor.dtype),
    tensor.shape,
    tensor.dtype,
  );
}

/**
 * Apply loaded checkpoint weights to an existing model.
 *
 * Validation happens before mutation. Replacement arrays are staged first,
 * then installed only after the key set, shapes, and dtypes all match.
 */
export function applyCheckpoint(model: GPT, checkpoint: CheckpointData): void {
  const current = currentParameterMap(model);
  assertCheckpointMatchesModel(current, checkpoint.parameters);

  const stagedEntries: [string[], MxArray][] = [];
  const stagedArrays: MxArray[] = [];

  try {
    for (const [key, { path }] of current.entries()) {
      const tensor = checkpoint.parameters[key];
      if (tensor === undefined) {
        throw new Error(`applyCheckpoint: missing checkpoint parameter "${key}"`);
      }

      const nextArray = createCheckpointArray(tensor);
      stagedArrays.push(nextArray);
      stagedEntries.push([path, nextArray]);
    }

    mxEval(...stagedArrays);
    model.update(treeUnflatten(stagedEntries));
  } catch (error) {
    for (const array of stagedArrays) {
      array.free();
    }
    throw error;
  }

  for (const { value } of current.values()) {
    value.free();
  }
}

/** Recreate an AdamW optimizer from serialized checkpoint state. */
export function restoreAdamWFromCheckpoint(checkpoint: AdamWOptimizerCheckpoint): AdamW {
  const optimizer = new AdamW({
    learningRate: checkpoint.lr,
    beta1: checkpoint.beta1,
    beta2: checkpoint.beta2,
    eps: checkpoint.eps,
    weightDecay: checkpoint.weightDecay,
  });
  const state: Record<string, Record<string, MxArray>> = {};
  const stagedArrays: MxArray[] = [];

  try {
    for (const [path, slots] of Object.entries(checkpoint.state)) {
      const slotState: Record<string, MxArray> = {};
      for (const [slotName, tensor] of Object.entries(slots)) {
        const array = createCheckpointArray(tensor);
        stagedArrays.push(array);
        slotState[slotName] = array;
      }
      state[path] = slotState;
    }

    mxEval(...stagedArrays);
    optimizer.restore({
      kind: "adamw",
      step: checkpoint.step,
      lr: checkpoint.lr,
      beta1: checkpoint.beta1,
      beta2: checkpoint.beta2,
      eps: checkpoint.eps,
      weightDecay: checkpoint.weightDecay,
      state,
    });
    return optimizer;
  } catch (error) {
    for (const array of stagedArrays) {
      array.free();
    }
    optimizer[Symbol.dispose]();
    throw error;
  }
}
