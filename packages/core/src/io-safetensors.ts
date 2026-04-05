/**
 * Safetensors load/save helpers.
 * @module
 */

import { closeSync, mkdirSync, openSync, readSync, rmSync, writeSync } from "fs";
import { dirname } from "path";

import type { MxArray } from "./array";
import {
  buildTensorBytes,
  byteRangeForTensor,
  createTensorFromBytes,
  DTYPE_BYTE_SIZE,
  DTYPE_TO_SAFETENSORS,
  findManifestEntry,
  type LoadedSafetensors,
  loadTensorFromManifestEntry,
  readManifest,
  readTensorBytes,
  type SafetensorTensorChunkEntry,
  type SafetensorTensorEntry,
  type SupportedSafetensorsDType,
  tensorElementCount,
  toSupportedSafetensorsDType,
} from "./io-safetensors-format";

type ChunkIterable = Iterable<Uint8Array> | AsyncIterable<Uint8Array>;

export type SafetensorByteChunkEntry = {
  name: string;
  startByte: number;
  byteLength: number;
  bytes: Uint8Array;
};

export type SafetensorTensorInfo = {
  name: string;
  shape: number[];
  dtype: SupportedSafetensorsDType;
  byteLength: number;
};

export type InspectedSafetensors = {
  metadata: Record<string, string>;
  tensors: SafetensorTensorInfo[];
};

export type SafetensorWriteEntry = {
  name: string;
  shape: number[];
  dtype: SupportedSafetensorsDType;
  chunks: () => ChunkIterable;
};

function writeAll(fileDescriptor: number, bytes: Uint8Array): void {
  let written = 0;
  while (written < bytes.byteLength) {
    const chunkSize = writeSync(fileDescriptor, bytes, written, bytes.byteLength - written);
    if (chunkSize <= 0) {
      throw new Error("saveSafetensors: failed to write the full tensor payload.");
    }
    written += chunkSize;
  }
}

function sortedWriteEntries(
  entries: readonly SafetensorWriteEntry[],
): Array<SafetensorWriteEntry & { expectedBytes: number }> {
  return [...entries]
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((entry) => ({
      ...entry,
      expectedBytes: tensorElementCount(entry.shape) * DTYPE_BYTE_SIZE[entry.dtype],
    }));
}

function writeSafetensorsHeader(
  fileDescriptor: number,
  entries: readonly (SafetensorWriteEntry & { expectedBytes: number })[],
  metadata: Record<string, string>,
): void {
  const headerEntries: [string, unknown][] = [];
  let dataOffset = 0;

  for (const entry of entries) {
    const nextOffset = dataOffset + entry.expectedBytes;
    headerEntries.push([
      entry.name,
      {
        dtype: DTYPE_TO_SAFETENSORS[entry.dtype],
        shape: entry.shape,
        data_offsets: [dataOffset, nextOffset],
      },
    ]);
    dataOffset = nextOffset;
  }

  const header: Record<string, unknown> = Object.fromEntries(headerEntries);
  if (Object.keys(metadata).length > 0) {
    header.__metadata__ = metadata;
  }

  const encodedHeader = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(encodedHeader.byteLength), true);
  writeAll(fileDescriptor, prefix);
  writeAll(fileDescriptor, encodedHeader);
}

async function writeSafetensorsEntries(
  fileDescriptor: number,
  entries: readonly (SafetensorWriteEntry & { expectedBytes: number })[],
): Promise<void> {
  for (const entry of entries) {
    let written = 0;
    for await (const chunk of entry.chunks()) {
      written += chunk.byteLength;
      writeAll(fileDescriptor, chunk);
    }

    if (written !== entry.expectedBytes) {
      throw new Error(
        `saveSafetensorsStream: entry "${entry.name}" wrote ${written} bytes, expected ${entry.expectedBytes}.`,
      );
    }
  }
}

async function* singleChunk(bytes: Uint8Array): AsyncGenerator<Uint8Array, void, void> {
  yield bytes;
}

/**
 * Iterate tensors in a safetensors file without materializing the entire file at once.
 */
export async function* iterateSafetensors(
  path: string,
  options: { include?: (name: string) => boolean } = {},
): AsyncGenerator<SafetensorTensorEntry, void, void> {
  const manifest = await readManifest(path);
  const include = options.include ?? (() => true);

  for (const entry of manifest.tensorHeaders) {
    if (!include(entry.name)) {
      continue;
    }

    yield {
      name: entry.name,
      tensor: await loadTensorFromManifestEntry(manifest, entry.name, entry.header),
    };
  }
}

/** Inspect tensor metadata in a safetensors file without loading tensor payloads. */
export async function inspectSafetensors(path: string): Promise<InspectedSafetensors> {
  const manifest = await readManifest(path);
  return {
    metadata: manifest.metadata,
    tensors: manifest.tensorHeaders.map((entry) => {
      const range = byteRangeForTensor(manifest, entry.name, entry.header);
      return {
        name: entry.name,
        shape: [...entry.header.shape],
        dtype: range.dtype,
        byteLength: range.byteLength,
      };
    }),
  };
}

/**
 * Iterate a single tensor in bounded first-axis chunks.
 *
 * This is intended for exceptional large tensors that should not be bridged
 * through one giant JavaScript typed array in a single step.
 */
export async function* iterateSafetensorTensorChunks(
  path: string,
  name: string,
  options: { maxBytesPerChunk?: number } = {},
): AsyncGenerator<SafetensorTensorChunkEntry, void, void> {
  const manifest = await readManifest(path);
  const entry = findManifestEntry(manifest, name);
  const tensorRange = byteRangeForTensor(manifest, name, entry.header);
  const [firstAxis = 1, ...restShape] = entry.header.shape;

  if (entry.header.shape.length === 0 || firstAxis <= 0) {
    yield {
      name,
      startIndex: 0,
      elementCount: firstAxis,
      tensor: await loadTensorFromManifestEntry(manifest, name, entry.header),
    };
    return;
  }

  const rowByteLength = tensorElementCount(restShape) * DTYPE_BYTE_SIZE[tensorRange.dtype];
  if (rowByteLength <= 0) {
    yield {
      name,
      startIndex: 0,
      elementCount: firstAxis,
      tensor: await loadTensorFromManifestEntry(manifest, name, entry.header),
    };
    return;
  }

  const maxBytesPerChunk = options.maxBytesPerChunk ?? tensorRange.byteLength;
  const rowsPerChunk = Math.max(1, Math.floor(maxBytesPerChunk / rowByteLength));

  for (let startIndex = 0; startIndex < firstAxis; startIndex += rowsPerChunk) {
    const elementCount = Math.min(rowsPerChunk, firstAxis - startIndex);
    const absoluteStart = tensorRange.absoluteStart + startIndex * rowByteLength;
    const absoluteEnd = absoluteStart + elementCount * rowByteLength;
    const bytes = await readTensorBytes(manifest.file, absoluteStart, absoluteEnd);
    yield {
      name,
      startIndex,
      elementCount,
      tensor: createTensorFromBytes(bytes, [elementCount, ...restShape], tensorRange.dtype),
    };
  }
}

/**
 * Iterate raw tensor bytes in bounded chunks without bridging them through MLX arrays.
 */
export async function* iterateSafetensorByteChunks(
  path: string,
  name: string,
  options: { maxBytesPerChunk?: number } = {},
): AsyncGenerator<SafetensorByteChunkEntry, void, void> {
  const manifest = await readManifest(path);
  const entry = findManifestEntry(manifest, name);
  const tensorRange = byteRangeForTensor(manifest, name, entry.header);
  const maxBytesPerChunk = options.maxBytesPerChunk ?? tensorRange.byteLength;
  const fileDescriptor = openSync(path, "r");

  try {
    for (
      let absoluteStart = tensorRange.absoluteStart;
      absoluteStart < tensorRange.absoluteEnd;
      absoluteStart += maxBytesPerChunk
    ) {
      const absoluteEnd = Math.min(tensorRange.absoluteEnd, absoluteStart + maxBytesPerChunk);
      const bytes = new Uint8Array(absoluteEnd - absoluteStart);
      let read = 0;

      while (read < bytes.byteLength) {
        const chunkSize = readSync(
          fileDescriptor,
          bytes,
          read,
          bytes.byteLength - read,
          absoluteStart + read,
        );
        if (chunkSize <= 0) {
          throw new Error(
            `iterateSafetensorByteChunks: failed to read the full byte range for "${name}".`,
          );
        }
        read += chunkSize;
      }

      yield {
        name,
        startByte: absoluteStart - tensorRange.absoluteStart,
        byteLength: bytes.byteLength,
        bytes,
      };
    }
  } finally {
    closeSync(fileDescriptor);
  }
}

/** Bridge a tensor into safetensors-compatible bytes. */
export function tensorBytes(
  tensor: MxArray,
  dtype: SupportedSafetensorsDType = toSupportedSafetensorsDType(tensor.dtype),
): Uint8Array {
  return buildTensorBytes(tensor, dtype);
}

/** Save a safetensors file from streaming byte-producing entries. */
export async function saveSafetensorsStream(
  entries: readonly SafetensorWriteEntry[],
  path: string,
  metadata: Record<string, string> = {},
): Promise<void> {
  const writableEntries = sortedWriteEntries(entries);
  mkdirSync(dirname(path), { recursive: true });
  const fileDescriptor = openSync(path, "w");

  try {
    writeSafetensorsHeader(fileDescriptor, writableEntries, metadata);
    await writeSafetensorsEntries(fileDescriptor, writableEntries);
  } catch (error) {
    closeSync(fileDescriptor);
    rmSync(path, { force: true });
    throw error;
  }

  closeSync(fileDescriptor);
}

/**
 * Save named tensors to a safetensors file.
 */
export async function saveSafetensors(
  tensors: Record<string, MxArray>,
  path: string,
  metadata: Record<string, string> = {},
): Promise<void> {
  const writableEntries: SafetensorWriteEntry[] = Object.entries(tensors).map(([key, tensor]) => {
    const dtype = toSupportedSafetensorsDType(tensor.dtype);
    const shape = [...tensor.shape];
    return {
      name: key,
      shape,
      dtype,
      chunks: () => singleChunk(tensorBytes(tensor, dtype)),
    };
  });

  await saveSafetensorsStream(writableEntries, path, metadata);
}

/**
 * Load named tensors from a safetensors file.
 */
export async function loadSafetensors(path: string): Promise<LoadedSafetensors> {
  const manifest = await readManifest(path);
  const tensors: Record<string, MxArray> = {};

  try {
    for (const entry of manifest.tensorHeaders) {
      tensors[entry.name] = await loadTensorFromManifestEntry(manifest, entry.name, entry.header);
    }

    return { tensors, metadata: manifest.metadata };
  } catch (error) {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
    throw error;
  }
}

export type {
  LoadedSafetensors,
  SafetensorTensorChunkEntry,
  SafetensorTensorEntry,
  SupportedSafetensorsDType,
};
