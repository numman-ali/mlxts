/**
 * Shared safetensors parsing and tensor-bridge helpers.
 * @module
 */

import { MxArray, readResultArray } from "./array";
import { getDataPointer } from "./array-ffi-data";
import { defaultStream } from "./device";
import type { DType } from "./dtype";
import { DTYPE_BYTE_SIZE, DTYPE_TO_MLX } from "./dtype";
import { checkStatus } from "./error";
import { ffi, nativeSlice, ptr, unwrapPointer } from "./ffi";
import { formatShape } from "./format-shape";

export type SupportedSafetensorsDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "float16"
  | "bfloat16"
  | "float32"
  | "float64";

type SafetensorsDTypeTag =
  | "BOOL"
  | "U8"
  | "U16"
  | "U32"
  | "U64"
  | "I8"
  | "I16"
  | "I32"
  | "I64"
  | "F16"
  | "BF16"
  | "F32"
  | "F64";

type SafetensorsTensorHeader = {
  dtype: SafetensorsDTypeTag;
  shape: number[];
  data_offsets: [number, number];
};

type RawStorageView =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array;

export type LoadedSafetensors = {
  tensors: Record<string, MxArray>;
  metadata: Record<string, string>;
};

export type SafetensorTensorEntry = {
  name: string;
  tensor: MxArray;
};

export type SafetensorTensorChunkEntry = {
  name: string;
  startIndex: number;
  elementCount: number;
  tensor: MxArray;
};

const DTYPE_TO_SAFETENSORS = {
  bool: "BOOL",
  uint8: "U8",
  uint16: "U16",
  uint32: "U32",
  uint64: "U64",
  int8: "I8",
  int16: "I16",
  int32: "I32",
  int64: "I64",
  float16: "F16",
  bfloat16: "BF16",
  float32: "F32",
  float64: "F64",
} as const satisfies Record<SupportedSafetensorsDType, SafetensorsDTypeTag>;

const SAFETENSORS_TO_DTYPE: Record<SafetensorsDTypeTag, SupportedSafetensorsDType> = {
  BOOL: "bool",
  U8: "uint8",
  U16: "uint16",
  U32: "uint32",
  U64: "uint64",
  I8: "int8",
  I16: "int16",
  I32: "int32",
  I64: "int64",
  F16: "float16",
  BF16: "bfloat16",
  F32: "float32",
  F64: "float64",
};

function isSafetensorsHeaderRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafetensorsDTypeTag(value: string): value is SafetensorsDTypeTag {
  return value in SAFETENSORS_TO_DTYPE;
}

export function toSupportedSafetensorsDType(dtype: DType): SupportedSafetensorsDType {
  switch (dtype) {
    case "bool":
    case "uint8":
    case "uint16":
    case "uint32":
    case "uint64":
    case "int8":
    case "int16":
    case "int32":
    case "int64":
    case "float16":
    case "bfloat16":
    case "float32":
    case "float64":
      return dtype;
    default:
      throw new Error(
        `saveSafetensors: dtype ${dtype} is not supported by the safetensors bridge.`,
      );
  }
}

export function tensorElementCount(shape: readonly number[]): number {
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

function parseHeaderBytes(bytes: Uint8Array): Record<string, unknown> {
  const headerText = new TextDecoder().decode(bytes);
  const parsed: unknown = JSON.parse(headerText);
  if (!isSafetensorsHeaderRecord(parsed)) {
    throw new Error("loadSafetensors: expected the safetensors header to be a JSON object");
  }

  return Object.fromEntries(Object.entries(parsed));
}

async function readHeader(
  file: Bun.BunFile,
): Promise<{ header: Record<string, unknown>; dataStart: number; fileSize: number }> {
  const fileSize = file.size;
  const prefix = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  const headerLength = readHeaderLength(prefix);
  const dataStart = 8 + headerLength;
  if (fileSize < dataStart) {
    throw new Error("loadSafetensors: header length exceeds file size");
  }

  const headerBytes = new Uint8Array(await file.slice(8, dataStart).arrayBuffer());
  return { header: parseHeaderBytes(headerBytes), dataStart, fileSize };
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

export type SafetensorsManifest = {
  file: Bun.BunFile;
  fileSize: number;
  dataStart: number;
  metadata: Record<string, string>;
  tensorHeaders: Array<{ name: string; header: SafetensorsTensorHeader }>;
};

export type SafetensorByteRange = {
  dtype: SupportedSafetensorsDType;
  byteLength: number;
  absoluteStart: number;
  absoluteEnd: number;
};

function makeContiguous(tensor: MxArray): MxArray {
  return readResultArray("contiguous", (out) => {
    checkStatus(ffi.mlx_contiguous(out, tensor._ctx, false, defaultStream()), "contiguous");
  });
}

export function buildTensorBytes(tensor: MxArray, dtype: SupportedSafetensorsDType): Uint8Array {
  using contiguous = makeContiguous(tensor);
  contiguous.eval();
  const byteLength = contiguous.size * DTYPE_BYTE_SIZE[dtype];
  return new Uint8Array(nativeSlice(getDataPointer(contiguous, dtype), 0, byteLength).slice(0));
}

function createStorageView(bytes: Uint8Array, dtype: SupportedSafetensorsDType): RawStorageView {
  switch (dtype) {
    case "bool":
    case "uint8":
      return bytes;
    case "uint16":
    case "float16":
    case "bfloat16":
      return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    case "uint32":
      return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    case "uint64":
      return new BigUint64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
    case "int8":
      return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    case "int16":
      return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    case "int32":
      return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    case "int64":
      return new BigInt64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
    case "float32":
      return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    case "float64":
      return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
  }
}

export function createTensorFromBytes(
  bytes: Uint8Array,
  shape: number[],
  dtype: SupportedSafetensorsDType,
): MxArray {
  const storage = createStorageView(bytes, dtype);
  const shapeBuffer = shape.length === 0 ? new Int32Array(1) : new Int32Array(shape);
  return MxArray._fromCtx(
    unwrapPointer(
      ffi.mlx_array_new_data(ptr(storage), ptr(shapeBuffer), shape.length, DTYPE_TO_MLX[dtype]),
      "mlx_array_new_data",
    ),
  );
}

export function concatChunks(chunks: readonly Uint8Array[], totalLength: number): Uint8Array {
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

export async function readManifest(path: string): Promise<SafetensorsManifest> {
  const file = Bun.file(path);
  const { header, dataStart, fileSize } = await readHeader(file);
  const tensorHeaders: Array<{ name: string; header: SafetensorsTensorHeader }> = [];

  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      continue;
    }
    tensorHeaders.push({ name: key, header: assertTensorHeader(key, value) });
  }

  return {
    file,
    fileSize,
    dataStart,
    metadata: assertMetadata(header.__metadata__),
    tensorHeaders,
  };
}

export function findManifestEntry(
  manifest: SafetensorsManifest,
  name: string,
): { name: string; header: SafetensorsTensorHeader } {
  const entry = manifest.tensorHeaders.find((candidate) => candidate.name === name);
  if (entry === undefined) {
    throw new Error(`loadSafetensors: tensor "${name}" was not found in the safetensors file`);
  }
  return entry;
}

export function byteRangeForTensor(
  manifest: SafetensorsManifest,
  name: string,
  tensorHeader: SafetensorsTensorHeader,
): SafetensorByteRange {
  const dtype = SAFETENSORS_TO_DTYPE[tensorHeader.dtype];
  const [start, end] = tensorHeader.data_offsets;
  const byteLength = end - start;
  const expectedBytes = tensorElementCount(tensorHeader.shape) * DTYPE_BYTE_SIZE[dtype];
  if (byteLength !== expectedBytes) {
    throw new Error(
      `loadSafetensors: tensor "${name}" has ${byteLength} bytes in the header, expected ${expectedBytes} for ${dtype} ${formatShape(tensorHeader.shape)}`,
    );
  }

  const absoluteStart = manifest.dataStart + start;
  const absoluteEnd = manifest.dataStart + end;
  if (absoluteEnd > manifest.fileSize) {
    throw new Error(`loadSafetensors: tensor "${name}" extends past the end of the file`);
  }

  return {
    dtype,
    byteLength,
    absoluteStart,
    absoluteEnd,
  };
}

export async function readTensorBytes(
  file: Bun.BunFile,
  absoluteStart: number,
  absoluteEnd: number,
): Promise<Uint8Array> {
  return new Uint8Array(await file.slice(absoluteStart, absoluteEnd).arrayBuffer());
}

export async function loadTensorFromManifestEntry(
  manifest: SafetensorsManifest,
  name: string,
  tensorHeader: SafetensorsTensorHeader,
): Promise<MxArray> {
  const range = byteRangeForTensor(manifest, name, tensorHeader);
  const tensorBytes = await readTensorBytes(manifest.file, range.absoluteStart, range.absoluteEnd);
  return createTensorFromBytes(tensorBytes, tensorHeader.shape, range.dtype);
}

export { DTYPE_BYTE_SIZE, DTYPE_TO_SAFETENSORS };
