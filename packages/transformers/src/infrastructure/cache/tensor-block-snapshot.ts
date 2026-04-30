import { concatenate, type MxArray, retainArray, slice } from "@mlxts/core";

import type {
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "../../types";
import type { CacheLayerKind } from "./layer-kind";
import {
  cloneCacheArray,
  createLayerStateSnapshot,
  disposeLayerStateSnapshot,
  type LayerStateSnapshot,
} from "./runtime";
import {
  type CacheFactory,
  CacheSnapshot,
  type CacheSnapshotOptions,
  type SnapshotRestoreTarget,
  validateSnapshotOffset,
} from "./snapshot";

const FULL_KV_BLOCK_SIZE = 64;

function rank4Shape(
  tensor: MxArray,
  context: string,
): {
  batch: number;
  heads: number;
  length: number;
  width: number;
} {
  const [batch, heads, length, width] = tensor.shape;
  if (batch === undefined || heads === undefined || length === undefined || width === undefined) {
    throw new Error(`${context}: expected rank-4 cache tensors.`);
  }
  return { batch, heads, length, width };
}

function sliceCacheTensor(tensor: MxArray, start: number, stop: number, context: string): MxArray {
  const { batch, heads, length, width } = rank4Shape(tensor, context);
  if (start < 0 || stop < start || stop > length) {
    throw new Error(`${context}: slice [${start}, ${stop}) is out of range for length ${length}.`);
  }
  if (start === 0 && stop === length) {
    return retainArray(tensor);
  }
  return slice(tensor, [0, 0, start, 0], [batch, heads, stop, width]);
}

function blockAlignedLength(length: number, blockSize: number): number {
  return Math.floor(length / blockSize) * blockSize;
}

function tensorBlockLength(block: FullKVTensorBlock): number {
  return rank4Shape(block.array, "FullKVBlockSnapshot").length;
}

class FullKVTensorBlock {
  readonly array: MxArray;
  readonly byteSize: number;
  readonly #snapshotOwners = new Set<FullKVBlockSnapshot>();
  #chargedSnapshot: FullKVBlockSnapshot | null = null;
  #references = 0;

  constructor(array: MxArray) {
    this.array = array;
    this.byteSize = array.nbytes;
  }

  retainBorrower(): void {
    this.#references += 1;
  }

  releaseBorrower(): void {
    this.releaseReference();
  }

  retainSnapshot(snapshot: FullKVBlockSnapshot): void {
    this.#references += 1;
    this.#snapshotOwners.add(snapshot);
    if (this.#chargedSnapshot === null) {
      this.#chargedSnapshot = snapshot;
    }
  }

  releaseSnapshot(snapshot: FullKVBlockSnapshot): void {
    if (!this.#snapshotOwners.has(snapshot)) {
      return;
    }
    this.#snapshotOwners.delete(snapshot);
    if (this.#chargedSnapshot === snapshot) {
      const nextOwner = this.#snapshotOwners.values().next();
      this.#chargedSnapshot = nextOwner.done === true ? null : nextOwner.value;
    }
    this.releaseReference();
  }

  chargedByteSize(snapshot: FullKVBlockSnapshot): number {
    return this.#chargedSnapshot === snapshot ? this.byteSize : 0;
  }

  private releaseReference(): void {
    this.#references -= 1;
    if (this.#references < 0) {
      throw new Error("FullKVTensorBlock: released more references than were retained.");
    }
    if (this.#references > 0) {
      return;
    }
    this.array.free();
    this.#chargedSnapshot = null;
  }
}

type FullKVLayerBlocks = {
  keys: FullKVTensorBlock[];
  values: FullKVTensorBlock[];
  length: number;
  cursor: number;
};

function createTensorBlock(tensor: MxArray, start: number, stop: number): FullKVTensorBlock {
  using blockView = sliceCacheTensor(tensor, start, stop, "FullKVBlockSnapshot");
  return new FullKVTensorBlock(cloneCacheArray(blockView));
}

function releaseBorrowedBlocks(blocks: readonly FullKVTensorBlock[]): void {
  for (const block of blocks) {
    block.releaseBorrower();
  }
}

function retainBorrowedLayerBlocks(
  layer: FullKVLayerBlocks,
  offset: number,
  blockSize: number,
): FullKVLayerBlocks {
  const reusableLength = Math.min(layer.length, blockAlignedLength(offset, blockSize));
  const blockCount = Math.floor(reusableLength / blockSize);
  const keys = layer.keys.slice(0, blockCount);
  const values = layer.values.slice(0, blockCount);
  try {
    for (const block of keys) {
      block.retainBorrower();
    }
    for (const block of values) {
      block.retainBorrower();
    }
    return { keys, values, length: reusableLength, cursor: 0 };
  } catch (error) {
    releaseBorrowedBlocks(keys);
    releaseBorrowedBlocks(values);
    throw error;
  }
}

function releaseSnapshotBlocks(
  blocks: readonly FullKVTensorBlock[],
  snapshot: FullKVBlockSnapshot,
): void {
  for (const block of blocks) {
    block.releaseSnapshot(snapshot);
  }
}

function retainedTensorPieces(blocks: readonly FullKVTensorBlock[], length: number): MxArray[] {
  const pieces: MxArray[] = [];
  let remaining = length;
  for (const block of blocks) {
    if (remaining <= 0) {
      break;
    }
    const blockLength = tensorBlockLength(block);
    const takeLength = Math.min(blockLength, remaining);
    pieces.push(sliceCacheTensor(block.array, 0, takeLength, "FullKVBlockSnapshot.fork"));
    remaining -= takeLength;
  }
  if (remaining > 0) {
    for (const piece of pieces) {
      piece.free();
    }
    throw new Error("FullKVBlockSnapshot.fork: retained tensor blocks are shorter than requested.");
  }
  return pieces;
}

function concatenatePieces(pieces: MxArray[]): MxArray {
  const first = pieces[0];
  if (first === undefined) {
    throw new Error("FullKVBlockSnapshot.fork: expected at least one tensor block.");
  }
  if (pieces.length === 1) {
    return first;
  }
  return concatenate(pieces, 2);
}

function materializeTensorBlocks(
  blocks: readonly FullKVTensorBlock[],
  length: number,
): MxArray | null {
  if (length <= 0) {
    return null;
  }
  const pieces = retainedTensorPieces(blocks, length);
  let result: MxArray | null = null;
  try {
    result = concatenatePieces(pieces);
    return result;
  } finally {
    for (const piece of pieces) {
      if (piece !== result) {
        piece.free();
      }
    }
  }
}

function cloneSnapshotBlocks(
  tensor: MxArray,
  start: number,
  length: number,
  snapshot: FullKVBlockSnapshot,
  blockSize: number,
): FullKVTensorBlock[] {
  const blocks: FullKVTensorBlock[] = [];
  for (let blockStart = start; blockStart < length; blockStart += blockSize) {
    const blockStop = Math.min(length, blockStart + blockSize);
    const block = createTensorBlock(tensor, blockStart, blockStop);
    block.retainSnapshot(snapshot);
    blocks.push(block);
  }
  return blocks;
}

function retainSourceBlocks(
  blocks: readonly FullKVTensorBlock[],
  count: number,
  snapshot: FullKVBlockSnapshot,
): FullKVTensorBlock[] {
  const retained: FullKVTensorBlock[] = [];
  for (let index = 0; index < count; index += 1) {
    const block = blocks[index];
    if (block === undefined) {
      break;
    }
    block.retainSnapshot(snapshot);
    retained.push(block);
  }
  return retained;
}

/** Borrowed full-KV block source retained by cache forks between snapshots. */
export class FullKVBlockSnapshotSource implements Disposable {
  readonly #layers: FullKVLayerBlocks[];
  #disposed = false;

  constructor(layers: readonly FullKVLayerBlocks[], offset: number, blockSize: number) {
    const retainedLayers: FullKVLayerBlocks[] = [];
    try {
      for (const layer of layers) {
        retainedLayers.push(retainBorrowedLayerBlocks(layer, offset, blockSize));
      }
    } catch (error) {
      for (const layer of retainedLayers) {
        releaseBorrowedBlocks(layer.keys);
        releaseBorrowedBlocks(layer.values);
      }
      throw error;
    }
    this.#layers = retainedLayers;
  }

  layer(layerIndex: number): FullKVLayerBlocks | undefined {
    return this.#layers[layerIndex];
  }

  /** Retain this lineage across a cache ownership boundary. */
  retain(): FullKVBlockSnapshotSource {
    return new FullKVBlockSnapshotSource(this.#layers, Number.MAX_SAFE_INTEGER, FULL_KV_BLOCK_SIZE);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    for (const layer of this.#layers) {
      releaseBorrowedBlocks(layer.keys);
      releaseBorrowedBlocks(layer.values);
    }
    this.#disposed = true;
  }
}

class FullKVBlockSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly layerKinds: readonly CacheLayerKind[];
  readonly trimmable: boolean;
  readonly #layers: FullKVLayerBlocks[] = [];
  readonly #createCache: CacheFactory;
  #disposed = false;

  constructor(options: CacheSnapshotOptions<FullKVBlockSnapshotSource>) {
    this.offset = options.offset;
    this.layerKinds = [...options.layerKinds];
    this.trimmable = true;
    this.#createCache = options.createCache;
    try {
      for (let layerIndex = 0; layerIndex < options.layers.length; layerIndex += 1) {
        const layer = options.layers[layerIndex];
        if (layer === undefined) {
          continue;
        }
        this.#layers.push(this.createLayerBlocks(layer, options.source?.layer(layerIndex)));
      }
    } catch (error) {
      this[Symbol.dispose]();
      throw error;
    } finally {
      for (const layer of options.layers) {
        disposeLayerStateSnapshot(layer);
      }
    }
  }

  get estimatedByteSize(): number {
    let total = 0;
    for (const layer of this.#layers) {
      for (const block of layer.keys) {
        total += block.chargedByteSize(this);
      }
      for (const block of layer.values) {
        total += block.chargedByteSize(this);
      }
    }
    return total;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    if (this.#disposed) {
      return false;
    }
    const targetOffset = options.offset ?? this.offset;
    return validateSnapshotOffset(targetOffset, this.offset);
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    const targetOffset = options.offset ?? this.offset;
    if (!this.canFork({ offset: targetOffset })) {
      throw new Error(
        `TransformerCacheSnapshot.fork: cannot fork offset ${targetOffset} from snapshot offset ${this.offset}.`,
      );
    }

    const cache = this.#createCache();
    try {
      for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
        this.applyLayer(cache, layerIndex, targetOffset);
      }
      cache.advance(targetOffset);
      if (hasFullKVBlockSourceTarget(cache)) {
        const source = new FullKVBlockSnapshotSource(
          this.#layers,
          targetOffset,
          FULL_KV_BLOCK_SIZE,
        );
        cache.setFullKVBlockSnapshotSource(source);
      }
      return cache;
    } catch (error) {
      cache[Symbol.dispose]();
      throw error;
    }
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    for (const layer of this.#layers) {
      releaseSnapshotBlocks(layer.keys, this);
      releaseSnapshotBlocks(layer.values, this);
    }
    this.#disposed = true;
  }

  private createLayerBlocks(
    layer: LayerStateSnapshot,
    source: FullKVLayerBlocks | undefined,
  ): FullKVLayerBlocks {
    if (layer.keys === null || layer.values === null || layer.length === 0) {
      return { keys: [], values: [], length: 0, cursor: 0 };
    }
    const reusableLength = Math.min(
      blockAlignedLength(layer.length, FULL_KV_BLOCK_SIZE),
      source?.length ?? 0,
    );
    const sourceBlockCount = reusableLength / FULL_KV_BLOCK_SIZE;
    const blocks = {
      keys: retainSourceBlocks(source?.keys ?? [], sourceBlockCount, this),
      values: retainSourceBlocks(source?.values ?? [], sourceBlockCount, this),
      length: layer.length,
      cursor: layer.cursor,
    };
    try {
      blocks.keys.push(
        ...cloneSnapshotBlocks(layer.keys, reusableLength, layer.length, this, FULL_KV_BLOCK_SIZE),
      );
      blocks.values.push(
        ...cloneSnapshotBlocks(
          layer.values,
          reusableLength,
          layer.length,
          this,
          FULL_KV_BLOCK_SIZE,
        ),
      );
      return blocks;
    } catch (error) {
      releaseSnapshotBlocks(blocks.keys, this);
      releaseSnapshotBlocks(blocks.values, this);
      throw error;
    }
  }

  private layerLengthForFork(layer: FullKVLayerBlocks, targetOffset: number): number {
    if (targetOffset === this.offset) {
      return layer.length;
    }
    return Math.min(layer.length, targetOffset);
  }

  private applyLayer(cache: SnapshotRestoreTarget, layerIndex: number, targetOffset: number): void {
    const layer = this.#layers[layerIndex];
    if (layer === undefined || layer.length === 0) {
      return;
    }
    const length = this.layerLengthForFork(layer, targetOffset);
    if (length <= 0) {
      return;
    }
    let keys: MxArray | null = null;
    let values: MxArray | null = null;
    try {
      keys = materializeTensorBlocks(layer.keys, length);
      values = materializeTensorBlocks(layer.values, length);
      if (keys === null || values === null) {
        return;
      }
      const cursor = targetOffset === this.offset ? layer.cursor : 0;
      const snapshot = createLayerStateSnapshot(keys, values, length, cursor);
      keys = null;
      values = null;
      try {
        cache.restoreLayerSnapshot(layerIndex, snapshot, length, snapshot.cursor);
      } finally {
        disposeLayerStateSnapshot(snapshot);
      }
    } finally {
      keys?.free();
      values?.free();
    }
  }
}

export type FullKVBlockSnapshotSourceTarget = SnapshotRestoreTarget & {
  setFullKVBlockSnapshotSource(source: FullKVBlockSnapshotSource | null): void;
};

export function hasFullKVBlockSourceTarget(
  cache: SnapshotRestoreTarget,
): cache is FullKVBlockSnapshotSourceTarget {
  return (
    "setFullKVBlockSnapshotSource" in cache &&
    typeof cache.setFullKVBlockSnapshotSource === "function"
  );
}

function canUseFullKVBlockSnapshot(
  options: CacheSnapshotOptions<FullKVBlockSnapshotSource>,
): boolean {
  return (
    options.trimPolicy === "prefix" &&
    options.layerKinds.length > 0 &&
    options.layerKinds.every((kind) => kind === "full")
  );
}

/** Create the strongest snapshot backend supported by the cache shape. */
export function createCacheSnapshot(
  options: CacheSnapshotOptions<FullKVBlockSnapshotSource>,
): TransformerCacheSnapshot {
  if (canUseFullKVBlockSnapshot(options)) {
    return new FullKVBlockSnapshot(options);
  }
  return new CacheSnapshot(options);
}
