/**
 * Shared transformer cache snapshot contracts.
 * @module
 */

import type {
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "../../types";
import type { CacheLayerKind } from "./layer-kind";
import {
  disposeLayerStateSnapshot,
  type LayerStateSnapshot,
  layerStateSnapshotByteSize,
} from "./runtime";

/** Fork policy for retained cache snapshots. */
export type SnapshotTrimPolicy = "exact" | "prefix";

/** Cache target that can accept retained per-layer snapshot state. */
export type SnapshotRestoreTarget = TransformerCache & {
  restoreLayerSnapshot(
    layerIndex: number,
    snapshot: LayerStateSnapshot,
    length: number,
    cursor: number,
  ): void;
};

/** Factory used by snapshots when producing disposable cache forks. */
export type CacheFactory = () => SnapshotRestoreTarget;

/** Shared construction payload for snapshot backends. */
export type CacheSnapshotOptions<Source = null> = {
  offset: number;
  layerKinds: readonly CacheLayerKind[];
  layers: LayerStateSnapshot[];
  createCache: CacheFactory;
  trimPolicy: SnapshotTrimPolicy;
  source?: Source | null;
};

/** Return whether a requested fork offset is inside the retained snapshot. */
export function validateSnapshotOffset(offset: number, snapshotOffset: number): boolean {
  return Number.isInteger(offset) && offset >= 0 && offset <= snapshotOffset;
}

/** Standard retained snapshot for cache shapes without block sharing. */
export class CacheSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly estimatedByteSize: number;
  readonly layerKinds: readonly CacheLayerKind[];
  readonly trimmable: boolean;
  readonly #layers: LayerStateSnapshot[];
  readonly #createCache: CacheFactory;
  readonly #trimPolicy: SnapshotTrimPolicy;
  #disposed = false;

  constructor(options: CacheSnapshotOptions<unknown>) {
    this.offset = options.offset;
    this.layerKinds = [...options.layerKinds];
    this.estimatedByteSize = options.layers.reduce(
      (total, layer) => total + layerStateSnapshotByteSize(layer),
      0,
    );
    this.trimmable = options.trimPolicy === "prefix";
    this.#layers = options.layers;
    this.#createCache = options.createCache;
    this.#trimPolicy = options.trimPolicy;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    if (this.#disposed) {
      return false;
    }
    const targetOffset = options.offset ?? this.offset;
    if (!validateSnapshotOffset(targetOffset, this.offset)) {
      return false;
    }
    return targetOffset === this.offset || this.#trimPolicy === "prefix";
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
      disposeLayerStateSnapshot(layer);
    }
    this.#disposed = true;
  }

  private layerLengthForFork(layer: LayerStateSnapshot, targetOffset: number): number {
    if (targetOffset === this.offset) {
      return layer.length;
    }
    return Math.min(layer.length, targetOffset);
  }

  private layerCursorForFork(layer: LayerStateSnapshot, targetOffset: number): number {
    return targetOffset === this.offset ? layer.cursor : 0;
  }

  private applyLayer(cache: SnapshotRestoreTarget, layerIndex: number, targetOffset: number): void {
    const layer = this.#layers[layerIndex];
    if (layer === undefined || layer.keys === null || layer.values === null) {
      return;
    }
    const length = this.layerLengthForFork(layer, targetOffset);
    if (length <= 0) {
      return;
    }

    cache.restoreLayerSnapshot(
      layerIndex,
      layer,
      length,
      this.layerCursorForFork(layer, targetOffset),
    );
  }
}
