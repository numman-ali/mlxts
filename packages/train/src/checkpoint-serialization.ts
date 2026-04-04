import { type DType, MxArray, mxEval, treeFlatten } from "@mlxts/core";
import type { AdamW } from "@mlxts/optimizers";

import { isSupportedCheckpointDType } from "./checkpoint-manifest";
import type {
  AdamWOptimizerManifest,
  CheckpointTensor,
  CheckpointTensorMeta,
  ParameterizedModel,
  SupportedCheckpointDType,
  SupportedTypedArray,
} from "./checkpoint-types";

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

function sortedParameterEntries(model: ParameterizedModel): [string[], MxArray][] {
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

  const serialized = entries.map(({ key, value }) => ({
    key,
    shape: [...value.shape],
    dtype: checkpointDType(value.dtype),
    data: typedArrayToBytes(value.toTypedArray()),
  }));

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

export function serializeParameters(model: ParameterizedModel): {
  bytes: Uint8Array;
  parameters: Record<string, CheckpointTensorMeta>;
} {
  return serializeTensorEntries(
    sortedParameterEntries(model).map(([path, value]) => ({ key: path.join("."), value })),
  );
}

export function serializeOptimizer(optimizer: AdamW): {
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

export function readTensorSlice(
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

export function loadOptimizerState(
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

export function createCheckpointArray(tensor: CheckpointTensor): MxArray {
  return MxArray.fromData(
    typedArrayFromBytes(tensor.data, tensor.dtype),
    tensor.shape,
    tensor.dtype,
  );
}

export function currentParameterEntries(model: ParameterizedModel): [string[], MxArray][] {
  return sortedParameterEntries(model);
}
