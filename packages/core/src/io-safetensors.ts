/**
 * Safetensors load/save helpers.
 * @module
 */

import type { MxArray } from "./array";
import {
  buildTensorBytes,
  byteRangeForTensor,
  concatChunks,
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
  tensorElementCount,
  toSupportedSafetensorsDType,
} from "./io-safetensors-format";

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
 * Save named tensors to a safetensors file.
 */
export async function saveSafetensors(
  tensors: Record<string, MxArray>,
  path: string,
  metadata: Record<string, string> = {},
): Promise<void> {
  const entries = Object.entries(tensors).sort(([left], [right]) => left.localeCompare(right));
  const dataChunks: Uint8Array[] = [];
  const headerEntries: [string, unknown][] = [];
  let dataOffset = 0;

  for (const [key, tensor] of entries) {
    const dtype = toSupportedSafetensorsDType(tensor.dtype);
    const shape = [...tensor.shape];
    const bytes = buildTensorBytes(tensor, dtype);
    const expectedBytes = tensorElementCount(shape) * DTYPE_BYTE_SIZE[dtype];
    if (bytes.byteLength !== expectedBytes) {
      throw new Error(
        `saveSafetensors: tensor "${key}" has ${bytes.byteLength} bytes, expected ${expectedBytes} for ${dtype}`,
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

export type { LoadedSafetensors, SafetensorTensorChunkEntry, SafetensorTensorEntry };
