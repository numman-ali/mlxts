import { readFileSync } from "fs";
import type { GgufHeader, GgufTensorInfo } from "./types";

type Cursor = {
  offset: number;
};

const GGUF_MAGIC = "GGUF";

const GGUF_VALUE_UINT8 = 0;
const GGUF_VALUE_INT8 = 1;
const GGUF_VALUE_UINT16 = 2;
const GGUF_VALUE_INT16 = 3;
const GGUF_VALUE_UINT32 = 4;
const GGUF_VALUE_INT32 = 5;
const GGUF_VALUE_FLOAT32 = 6;
const GGUF_VALUE_BOOL = 7;
const GGUF_VALUE_STRING = 8;
const GGUF_VALUE_ARRAY = 9;
const GGUF_VALUE_UINT64 = 10;
const GGUF_VALUE_INT64 = 11;
const GGUF_VALUE_FLOAT64 = 12;

function ensureAvailable(bytes: Uint8Array, cursor: Cursor, size: number, context: string): void {
  if (cursor.offset + size > bytes.byteLength) {
    throw new Error(`parseGgufHeader: ${context} exceeds file size`);
  }
}

function readU32(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 4, "u32 field");
  const value = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 4).getUint32(0, true);
  cursor.offset += 4;
  return value;
}

function readI32(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 4, "i32 field");
  const value = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 4).getInt32(0, true);
  cursor.offset += 4;
  return value;
}

function readU64(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 8, "u64 field");
  const value = Number(
    new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 8).getBigUint64(0, true),
  );
  cursor.offset += 8;
  if (!Number.isSafeInteger(value)) {
    throw new Error("parseGgufHeader: encountered a u64 value larger than Number.MAX_SAFE_INTEGER");
  }
  return value;
}

function readI64(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 8, "i64 field");
  const value = Number(
    new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 8).getBigInt64(0, true),
  );
  cursor.offset += 8;
  return value;
}

function readF32(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 4, "f32 field");
  const value = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 4).getFloat32(0, true);
  cursor.offset += 4;
  return value;
}

function readF64(bytes: Uint8Array, cursor: Cursor): number {
  ensureAvailable(bytes, cursor, 8, "f64 field");
  const value = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 8).getFloat64(0, true);
  cursor.offset += 8;
  return value;
}

function readString(bytes: Uint8Array, cursor: Cursor): string {
  const length = readU64(bytes, cursor);
  ensureAvailable(bytes, cursor, length, "string field");
  const value = new TextDecoder().decode(bytes.subarray(cursor.offset, cursor.offset + length));
  cursor.offset += length;
  return value;
}

function readArray(bytes: Uint8Array, cursor: Cursor): unknown[] {
  const itemType = readU32(bytes, cursor);
  const length = readU64(bytes, cursor);
  const values: unknown[] = [];
  for (let index = 0; index < length; index += 1) {
    values.push(readMetadataValue(bytes, cursor, itemType));
  }
  return values;
}

function readMetadataValue(bytes: Uint8Array, cursor: Cursor, type: number): unknown {
  switch (type) {
    case GGUF_VALUE_UINT8:
      ensureAvailable(bytes, cursor, 1, "u8 field");
      return bytes[cursor.offset++] ?? 0;
    case GGUF_VALUE_INT8:
      ensureAvailable(bytes, cursor, 1, "i8 field");
      return new DataView(bytes.buffer, bytes.byteOffset + cursor.offset++, 1).getInt8(0);
    case GGUF_VALUE_UINT16: {
      ensureAvailable(bytes, cursor, 2, "u16 field");
      const u16 = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 2).getUint16(
        0,
        true,
      );
      cursor.offset += 2;
      return u16;
    }
    case GGUF_VALUE_INT16: {
      ensureAvailable(bytes, cursor, 2, "i16 field");
      const i16 = new DataView(bytes.buffer, bytes.byteOffset + cursor.offset, 2).getInt16(0, true);
      cursor.offset += 2;
      return i16;
    }
    case GGUF_VALUE_UINT32:
      return readU32(bytes, cursor);
    case GGUF_VALUE_INT32:
      return readI32(bytes, cursor);
    case GGUF_VALUE_FLOAT32:
      return readF32(bytes, cursor);
    case GGUF_VALUE_BOOL:
      ensureAvailable(bytes, cursor, 1, "bool field");
      return (bytes[cursor.offset++] ?? 0) !== 0;
    case GGUF_VALUE_STRING:
      return readString(bytes, cursor);
    case GGUF_VALUE_ARRAY:
      return readArray(bytes, cursor);
    case GGUF_VALUE_UINT64:
      return readU64(bytes, cursor);
    case GGUF_VALUE_INT64:
      return readI64(bytes, cursor);
    case GGUF_VALUE_FLOAT64:
      return readF64(bytes, cursor);
    default:
      throw new Error(`parseGgufHeader: unsupported GGUF metadata value type ${type}`);
  }
}

function readTensorInfo(bytes: Uint8Array, cursor: Cursor): GgufTensorInfo {
  const name = readString(bytes, cursor);
  const dimensionCount = readU32(bytes, cursor);
  const dimensions: number[] = [];
  for (let index = 0; index < dimensionCount; index += 1) {
    dimensions.push(readU64(bytes, cursor));
  }
  const type = readU32(bytes, cursor);
  const offset = readU64(bytes, cursor);
  return { name, dimensions, type, offset };
}

/** Parse GGUF header metadata and tensor descriptors without reading tensor payloads. */
export function parseGgufHeader(path: string): GgufHeader {
  const bytes = new Uint8Array(readFileSync(path));
  if (bytes.byteLength < 24) {
    throw new Error("parseGgufHeader: file is too small to be a valid GGUF file");
  }

  const magic = new TextDecoder().decode(bytes.subarray(0, 4));
  if (magic !== GGUF_MAGIC) {
    throw new Error(`parseGgufHeader: expected magic "${GGUF_MAGIC}" but found "${magic}"`);
  }

  const cursor: Cursor = { offset: 4 };
  const version = readU32(bytes, cursor);
  const tensorCount = readU64(bytes, cursor);
  const metadataCount = readU64(bytes, cursor);
  const metadata: Record<string, unknown> = {};

  for (let index = 0; index < metadataCount; index += 1) {
    const key = readString(bytes, cursor);
    const type = readU32(bytes, cursor);
    metadata[key] = readMetadataValue(bytes, cursor, type);
  }

  const tensors: GgufTensorInfo[] = [];
  for (let index = 0; index < tensorCount; index += 1) {
    tensors.push(readTensorInfo(bytes, cursor));
  }

  return { version, metadata, tensors };
}
