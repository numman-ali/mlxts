/**
 * Tensor serialization helpers for ecosystem interop.
 *
 * This module currently focuses on safetensors because it is the simplest
 * portable weight format for TypeScript and Python tooling to share.
 *
 * @module
 */

import { MxArray } from "./array";
import { DTYPE_BYTE_SIZE, type DType } from "./dtype";
import { formatShape } from "./format-shape";

type SupportedSafetensorsDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";

type SafetensorsDTypeTag = "BOOL" | "U8" | "U16" | "U32" | "I8" | "I16" | "I32" | "F32" | "F64";

type SafetensorsTensorHeader = {
  dtype: SafetensorsDTypeTag;
  shape: number[];
  data_offsets: [number, number];
};

export type LoadedSafetensors = {
  tensors: Record<string, MxArray>;
  metadata: Record<string, string>;
};

const DTYPE_TO_SAFETENSORS = {
  bool: "BOOL",
  uint8: "U8",
  uint16: "U16",
  uint32: "U32",
  int8: "I8",
  int16: "I16",
  int32: "I32",
  float32: "F32",
  float64: "F64",
} as const satisfies Record<SupportedSafetensorsDType, SafetensorsDTypeTag>;

const SAFETENSORS_TO_DTYPE: Record<SafetensorsDTypeTag, SupportedSafetensorsDType> = {
  BOOL: "bool",
  U8: "uint8",
  U16: "uint16",
  U32: "uint32",
  I8: "int8",
  I16: "int16",
  I32: "int32",
  F32: "float32",
  F64: "float64",
};

function isSafetensorsHeaderRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafetensorsDTypeTag(value: string): value is SafetensorsDTypeTag {
  return value in SAFETENSORS_TO_DTYPE;
}

function toSupportedSafetensorsDType(dtype: DType): SupportedSafetensorsDType {
  switch (dtype) {
    case "bool":
    case "uint8":
    case "uint16":
    case "uint32":
    case "int8":
    case "int16":
    case "int32":
    case "float32":
    case "float64":
      return dtype;
    default:
      throw new Error(
        `saveSafetensors: dtype ${dtype} is not supported by the current TypeScript safetensors bridge.`,
      );
  }
}

function tensorElementCount(shape: readonly number[]): number {
  if (shape.length === 0) {
    return 1;
  }
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function readHeaderLength(bytes: Uint8Array): number {
  if (bytes.byteLength < 8) {
    throw new Error("loadSafetensors: file is too small to contain a valid header");
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2) {
    throw new Error(`loadSafetensors: invalid header length ${headerLength}`);
  }
  return headerLength;
}

function readHeader(bytes: Uint8Array): { header: Record<string, unknown>; dataStart: number } {
  const headerLength = readHeaderLength(bytes);
  const dataStart = 8 + headerLength;
  if (bytes.byteLength < dataStart) {
    throw new Error("loadSafetensors: header length exceeds file size");
  }

  const headerText = new TextDecoder().decode(bytes.subarray(8, dataStart));
  const parsed: unknown = JSON.parse(headerText);
  if (!isSafetensorsHeaderRecord(parsed)) {
    throw new Error("loadSafetensors: expected the safetensors header to be a JSON object");
  }

  return { header: Object.fromEntries(Object.entries(parsed)), dataStart };
}

function assertMetadata(value: unknown): Record<string, string> {
  if (value === undefined) {
    return {};
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("loadSafetensors: __metadata__ must be an object of strings");
  }

  const metadata: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry !== "string") {
      throw new Error(`loadSafetensors: metadata value for "${key}" must be a string`);
    }
    metadata[key] = entry;
  }
  return metadata;
}

function assertTensorHeader(key: string, value: unknown): SafetensorsTensorHeader {
  if (!isSafetensorsHeaderRecord(value)) {
    throw new Error(`loadSafetensors: header entry "${key}" must be an object`);
  }

  const dtype = value.dtype;
  const shape = value.shape;
  const dataOffsets = value.data_offsets;
  if (
    typeof dtype !== "string" ||
    !isSafetensorsDTypeTag(dtype) ||
    !Array.isArray(shape) ||
    !Array.isArray(dataOffsets) ||
    dataOffsets.length !== 2
  ) {
    throw new Error(`loadSafetensors: header entry "${key}" is malformed`);
  }

  const normalizedShape = shape.map((dimension, index) => {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(
        `loadSafetensors: shape[${index}] for "${key}" must be a non-negative integer`,
      );
    }
    return dimension;
  });

  const start = dataOffsets[0];
  const end = dataOffsets[1];
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
    throw new Error(`loadSafetensors: data_offsets for "${key}" must be an ordered integer pair`);
  }

  return {
    dtype,
    shape: normalizedShape,
    data_offsets: [start, end],
  };
}

function buildTensorBytes(tensor: MxArray): Uint8Array {
  const typed = tensor.toTypedArray();
  return new Uint8Array(typed.buffer.slice(typed.byteOffset, typed.byteOffset + typed.byteLength));
}

function concatChunks(chunks: readonly Uint8Array[], totalLength: number): Uint8Array {
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

/**
 * Save named tensors to a safetensors file.
 *
 * Only dtypes that round-trip cleanly through the current TypeScript bridge
 * are supported.
 */
export async function saveSafetensors(
  tensors: Record<string, MxArray>,
  path: string,
  metadata: Record<string, string> = {},
): Promise<void> {
  const entries = Object.entries(tensors).sort(([left], [right]) => left.localeCompare(right));
  const arrays = entries.map(([, tensor]) => tensor);
  if (arrays.length > 0) {
    for (const tensor of arrays) {
      tensor.eval();
    }
  }

  const dataChunks: Uint8Array[] = [];
  const headerEntries: [string, SafetensorsTensorHeader][] = [];
  let dataOffset = 0;

  for (const [key, tensor] of entries) {
    const dtype = toSupportedSafetensorsDType(tensor.dtype);
    const shape = [...tensor.shape];
    const expectedBytes = tensorElementCount(shape) * DTYPE_BYTE_SIZE[dtype];
    const bytes = buildTensorBytes(tensor);
    if (bytes.byteLength !== expectedBytes) {
      throw new Error(
        `saveSafetensors: tensor "${key}" has ${bytes.byteLength} bytes, expected ${expectedBytes} for ${dtype} ${formatShape(shape)}`,
      );
    }

    const nextOffset = dataOffset + bytes.byteLength;
    headerEntries.push([
      key,
      {
        dtype: DTYPE_TO_SAFETENSORS[dtype],
        shape,
        data_offsets: [dataOffset, nextOffset],
      },
    ]);
    dataChunks.push(bytes);
    dataOffset = nextOffset;
  }

  const header: Record<string, unknown> = Object.fromEntries(headerEntries);
  if (Object.keys(metadata).length > 0) {
    header.__metadata__ = metadata;
  }

  const encodedHeader = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(encodedHeader.byteLength), true);
  const payload = concatChunks(
    [prefix, encodedHeader, ...dataChunks],
    8 + encodedHeader.byteLength + dataOffset,
  );
  await Bun.write(path, payload);
}

/**
 * Load named tensors from a safetensors file.
 */
export async function loadSafetensors(path: string): Promise<LoadedSafetensors> {
  const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
  const { header, dataStart } = readHeader(bytes);
  const metadata = assertMetadata(header.__metadata__);
  const tensors: Record<string, MxArray> = {};

  try {
    for (const [key, value] of Object.entries(header)) {
      if (key === "__metadata__") {
        continue;
      }

      const tensorHeader = assertTensorHeader(key, value);
      const dtype = SAFETENSORS_TO_DTYPE[tensorHeader.dtype];
      const [start, end] = tensorHeader.data_offsets;
      const byteLength = end - start;
      const expectedBytes = tensorElementCount(tensorHeader.shape) * DTYPE_BYTE_SIZE[dtype];
      if (byteLength !== expectedBytes) {
        throw new Error(
          `loadSafetensors: tensor "${key}" has ${byteLength} bytes in the header, expected ${expectedBytes} for ${dtype} ${formatShape(tensorHeader.shape)}`,
        );
      }

      const absoluteStart = dataStart + start;
      const absoluteEnd = dataStart + end;
      if (absoluteEnd > bytes.byteLength) {
        throw new Error(`loadSafetensors: tensor "${key}" extends past the end of the file`);
      }

      const tensorBytes = bytes.subarray(absoluteStart, absoluteEnd);
      let typed:
        | Float32Array
        | Float64Array
        | Int8Array
        | Int16Array
        | Int32Array
        | Uint8Array
        | Uint16Array
        | Uint32Array;
      switch (dtype) {
        case "bool":
        case "uint8":
          typed = new Uint8Array(tensorBytes.slice().buffer);
          break;
        case "uint16":
          typed = new Uint16Array(tensorBytes.slice().buffer);
          break;
        case "uint32":
          typed = new Uint32Array(tensorBytes.slice().buffer);
          break;
        case "int8":
          typed = new Int8Array(tensorBytes.slice().buffer);
          break;
        case "int16":
          typed = new Int16Array(tensorBytes.slice().buffer);
          break;
        case "int32":
          typed = new Int32Array(tensorBytes.slice().buffer);
          break;
        case "float32":
          typed = new Float32Array(tensorBytes.slice().buffer);
          break;
        case "float64":
          typed = new Float64Array(tensorBytes.slice().buffer);
          break;
      }

      tensors[key] = MxArray.fromData(typed, tensorHeader.shape, dtype);
    }

    return { tensors, metadata };
  } catch (error) {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
    throw error;
  }
}
