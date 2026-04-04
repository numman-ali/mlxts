import { readFileSync } from "fs";
import { join } from "path";

import {
  type AdamWOptimizerManifest,
  CHECKPOINT_VERSION,
  type CheckpointManifest,
  type CheckpointTensorMeta,
  MANIFEST_FILENAME,
  type SupportedCheckpointDType,
} from "./checkpoint-types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context}: expected a finite number`);
  }
  return value;
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

export function isSupportedCheckpointDType(value: string): value is SupportedCheckpointDType {
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
  if (typeof value !== "string") {
    throw new Error(`${context}: expected a string`);
  }
  if (!isSupportedCheckpointDType(value)) {
    throw new Error(`${context}: unsupported checkpoint dtype "${value}"`);
  }
  return value;
}

export function bytesPerElement(dtype: SupportedCheckpointDType): number {
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
    throw new Error('checkpoint manifest optimizer.kind: expected "adamw"');
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

export function readManifest(path: string): CheckpointManifest {
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

  if (!("metadata" in parsed)) {
    throw new Error("checkpoint manifest metadata: expected a metadata field");
  }

  const kind = parsed.kind;
  if (kind !== "snapshot" && kind !== "resume" && kind !== "best") {
    throw new Error(`checkpoint manifest kind "${String(kind)}" is unsupported`);
  }

  const manifest: CheckpointManifest = {
    version: CHECKPOINT_VERSION,
    kind,
    metadata: parsed.metadata,
    step: readNumber(parsed.step, "checkpoint manifest step"),
    parameters: readParameters(parsed.parameters),
  };

  if (parsed.optimizer !== undefined) {
    manifest.optimizer = readOptimizer(parsed.optimizer);
  }
  if (manifest.kind === "resume" && manifest.optimizer === undefined) {
    throw new Error("checkpoint manifest: resume checkpoints require optimizer metadata");
  }
  if (
    (manifest.kind === "snapshot" || manifest.kind === "best") &&
    manifest.optimizer !== undefined
  ) {
    throw new Error(
      "checkpoint manifest: snapshot/best checkpoints must not include optimizer metadata",
    );
  }
  if (manifest.optimizer !== undefined && manifest.optimizer.step !== manifest.step) {
    throw new Error(
      `checkpoint manifest optimizer.step ${manifest.optimizer.step} does not match checkpoint step ${manifest.step}`,
    );
  }

  return manifest;
}

export function shiftTensorMeta(
  meta: CheckpointTensorMeta,
  baseOffset: number,
): CheckpointTensorMeta {
  return {
    ...meta,
    offset: meta.offset + baseOffset,
  };
}

export function shiftOptimizerOffsets(
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
