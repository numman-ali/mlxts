/**
 * Family-local hybrid cache for Qwen 3.5 text layers.
 * @module
 */

import { type MxArray, mxEval, retainArray, slice } from "@mlxts/core";

import { KVCache } from "../../infrastructure/cache";
import { cloneCacheArray } from "../../infrastructure/cache/runtime";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "../../infrastructure/cache/view";
import type {
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "../../types";
import type { Qwen3_5LayerType } from "./types";

export type Qwen3_5LinearLayerState = {
  convState: MxArray | null;
  recurrentState: MxArray | null;
};

type LayerCacheState =
  | {
      type: "full_attention";
      cache: KVCache;
    }
  | {
      type: "linear_attention";
      state: Qwen3_5LinearLayerState;
    };

type FullAttentionLayerSnapshot = {
  type: "full_attention";
  keys: MxArray | null;
  values: MxArray | null;
  length: number;
};

type LinearAttentionLayerSnapshot = {
  type: "linear_attention";
  convState: MxArray | null;
  recurrentState: MxArray | null;
};

type LayerCacheSnapshot = FullAttentionLayerSnapshot | LinearAttentionLayerSnapshot;

function assertLayerIndex(
  layers: readonly LayerCacheState[],
  layerIndex: number,
  context: string,
): LayerCacheState {
  const layer = layers[layerIndex];
  if (layer === undefined) {
    throw new Error(`${context}: layer ${layerIndex} is out of range.`);
  }
  return layer;
}

function disposeLinearState(state: Qwen3_5LinearLayerState): void {
  state.convState?.free();
  state.recurrentState?.free();
  state.convState = null;
  state.recurrentState = null;
}

function detachLinearState(state: MxArray | null): MxArray | null {
  return state === null ? null : retainArray(state);
}

function cloneOptionalState(state: MxArray | null): MxArray | null {
  return state === null ? null : cloneCacheArray(state);
}

function fullSnapshotLength(keys: MxArray): number {
  const length = keys.shape[2];
  if (length === undefined) {
    throw new Error("Qwen3_5TextCache.snapshot: expected rank-4 full-attention cache tensors.");
  }
  return length;
}

function cloneFullAttentionSnapshot(cache: KVCache): FullAttentionLayerSnapshot {
  const arrays = cache.arrays();
  let clonedKeys: MxArray | null = null;
  let clonedValues: MxArray | null = null;
  try {
    const keys = arrays[0];
    const values = arrays[1];
    if (keys === undefined && values === undefined) {
      return { type: "full_attention", keys: null, values: null, length: 0 };
    }
    if (keys === undefined || values === undefined || arrays.length !== 2) {
      throw new Error("Qwen3_5TextCache.snapshot: expected full-attention key/value tensors.");
    }
    clonedKeys = cloneCacheArray(keys);
    clonedValues = cloneCacheArray(values);
    return {
      type: "full_attention",
      keys: clonedKeys,
      values: clonedValues,
      length: fullSnapshotLength(keys),
    };
  } catch (error) {
    clonedKeys?.free();
    clonedValues?.free();
    throw error;
  } finally {
    for (const value of arrays) {
      value.free();
    }
  }
}

function cloneLinearAttentionSnapshot(
  state: Qwen3_5LinearLayerState,
): LinearAttentionLayerSnapshot {
  let convState: MxArray | null = null;
  let recurrentState: MxArray | null = null;
  try {
    convState = cloneOptionalState(state.convState);
    recurrentState = cloneOptionalState(state.recurrentState);
    return {
      type: "linear_attention",
      convState,
      recurrentState,
    };
  } catch (error) {
    convState?.free();
    recurrentState?.free();
    throw error;
  }
}

function disposeLayerCacheSnapshot(snapshot: LayerCacheSnapshot): void {
  if (snapshot.type === "full_attention") {
    snapshot.keys?.free();
    snapshot.values?.free();
    snapshot.keys = null;
    snapshot.values = null;
    snapshot.length = 0;
    return;
  }
  snapshot.convState?.free();
  snapshot.recurrentState?.free();
  snapshot.convState = null;
  snapshot.recurrentState = null;
}

function disposeLayerCacheSnapshots(snapshots: readonly LayerCacheSnapshot[]): void {
  for (const snapshot of snapshots) {
    disposeLayerCacheSnapshot(snapshot);
  }
}

function isValidSnapshotOffset(offset: number, snapshotOffset: number): boolean {
  return Number.isInteger(offset) && offset >= 0 && offset <= snapshotOffset;
}

function sliceFullAttentionSnapshotArray(value: MxArray, length: number): MxArray {
  const existingLength = value.shape[2];
  if (existingLength === undefined) {
    throw new Error("Qwen3_5TextCacheSnapshot.fork: expected rank-4 cache snapshot tensors.");
  }
  if (length === existingLength) {
    return retainArray(value);
  }
  const batch = value.shape[0] ?? 0;
  const heads = value.shape[1] ?? 0;
  const width = value.shape[3] ?? 0;
  return slice(value, [0, 0, 0, 0], [batch, heads, length, width]);
}

class Qwen3_5TextCacheSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly trimmable: boolean;
  readonly #layerTypes: Qwen3_5LayerType[];
  readonly #layers: LayerCacheSnapshot[];
  #disposed = false;

  constructor(options: {
    offset: number;
    layerTypes: Qwen3_5LayerType[];
    layers: LayerCacheSnapshot[];
  }) {
    this.offset = options.offset;
    this.#layerTypes = options.layerTypes;
    this.#layers = options.layers;
    this.trimmable = options.layerTypes.every((layerType) => layerType === "full_attention");
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    if (this.#disposed) {
      return false;
    }
    const targetOffset = options.offset ?? this.offset;
    if (!isValidSnapshotOffset(targetOffset, this.offset)) {
      return false;
    }
    return targetOffset === this.offset || this.trimmable;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    const targetOffset = options.offset ?? this.offset;
    if (!this.canFork({ offset: targetOffset })) {
      throw new Error(
        `Qwen3_5TextCacheSnapshot.fork: cannot fork offset ${targetOffset} from snapshot offset ${this.offset}.`,
      );
    }

    const cache = new Qwen3_5TextCache(this.#layerTypes);
    try {
      for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
        const layer = this.#layers[layerIndex];
        if (layer === undefined) {
          continue;
        }
        this.applyLayer(cache, layerIndex, layer, targetOffset);
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
    disposeLayerCacheSnapshots(this.#layers);
    this.#disposed = true;
  }

  private fullLengthForFork(layer: FullAttentionLayerSnapshot, targetOffset: number): number {
    if (targetOffset === this.offset) {
      return layer.length;
    }
    return Math.min(layer.length, targetOffset);
  }

  private applyLayer(
    cache: Qwen3_5TextCache,
    layerIndex: number,
    layer: LayerCacheSnapshot,
    targetOffset: number,
  ): void {
    if (layer.type === "linear_attention") {
      cache.updateLinearState(layerIndex, layer.convState, layer.recurrentState);
      return;
    }

    if (layer.keys === null || layer.values === null) {
      return;
    }
    const length = this.fullLengthForFork(layer, targetOffset);
    if (length <= 0) {
      return;
    }

    const keys = sliceFullAttentionSnapshotArray(layer.keys, length);
    const values = sliceFullAttentionSnapshotArray(layer.values, length);
    try {
      const updated = cache.updateAndFetch(layerIndex, keys, values);
      try {
        mxEval(updated.keys, updated.values);
      } finally {
        updated.keys.free();
        updated.values.free();
      }
    } finally {
      keys.free();
      values.free();
    }
  }
}

/** Hybrid text cache that mixes KV layers with recurrent linear-attention state. */
export class Qwen3_5TextCache implements TransformerCache {
  offset = 0;
  #layers: LayerCacheState[];

  constructor(layerTypes: readonly Qwen3_5LayerType[]) {
    if (layerTypes.length === 0) {
      throw new Error("Qwen3_5TextCache: layerTypes must contain at least one layer.");
    }
    this.#layers = layerTypes.map((layerType) =>
      layerType === "full_attention"
        ? {
            type: "full_attention",
            cache: new KVCache(1),
          }
        : {
            type: "linear_attention",
            state: {
              convState: null,
              recurrentState: null,
            },
          },
    );
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return this.#layers.every((layer) => layer.type === "full_attention");
  }

  snapshot(): TransformerCacheSnapshot {
    const snapshots: LayerCacheSnapshot[] = [];
    try {
      for (const layer of this.#layers) {
        snapshots.push(
          layer.type === "full_attention"
            ? cloneFullAttentionSnapshot(layer.cache)
            : cloneLinearAttentionSnapshot(layer.state),
        );
      }
      return new Qwen3_5TextCacheSnapshot({
        offset: this.offset,
        layerTypes: this.#layers.map((layer) => layer.type),
        layers: snapshots,
      });
    } catch (error) {
      disposeLayerCacheSnapshots(snapshots);
      throw error;
    }
  }

  advance(sequenceLength: number): void {
    if (!Number.isInteger(sequenceLength) || sequenceLength < 0) {
      throw new Error(
        `Qwen3_5TextCache.advance: sequenceLength must be a non-negative integer, got ${sequenceLength}.`,
      );
    }
    this.offset += sequenceLength;
  }

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    const layer = assertLayerIndex(this.#layers, layerIndex, "Qwen3_5TextCache.updateAndFetch");
    if (layer.type !== "full_attention") {
      throw new Error(
        `Qwen3_5TextCache.updateAndFetch: layer ${layerIndex} is ${layer.type}; KV updates only apply to full_attention layers.`,
      );
    }
    return layer.cache.updateAndFetch(0, keys, values);
  }

  [INTERNAL_CACHE_VIEW](layerIndex: number, keys: MxArray, values: MxArray): TransformerCacheView {
    const layer = assertLayerIndex(this.#layers, layerIndex, "Qwen3_5TextCache.cacheView");
    if (layer.type !== "full_attention") {
      throw new Error(
        `Qwen3_5TextCache.cacheView: layer ${layerIndex} is ${layer.type}; KV updates only apply to full_attention layers.`,
      );
    }
    return layer.cache[INTERNAL_CACHE_VIEW](0, keys, values);
  }

  arrays(): MxArray[] {
    const arrays: MxArray[] = [];
    for (const layer of this.#layers) {
      if (layer.type === "full_attention") {
        arrays.push(...layer.cache.arrays());
        continue;
      }
      if (layer.state.convState !== null) {
        arrays.push(retainArray(layer.state.convState));
      }
      if (layer.state.recurrentState !== null) {
        arrays.push(retainArray(layer.state.recurrentState));
      }
    }
    return arrays;
  }

  linearState(layerIndex: number): Qwen3_5LinearLayerState {
    const layer = assertLayerIndex(this.#layers, layerIndex, "Qwen3_5TextCache.linearState");
    if (layer.type !== "linear_attention") {
      throw new Error(
        `Qwen3_5TextCache.linearState: layer ${layerIndex} is ${layer.type}; linear state only exists on linear_attention layers.`,
      );
    }
    return layer.state;
  }

  updateLinearState(
    layerIndex: number,
    convState: MxArray | null,
    recurrentState: MxArray | null,
  ): void {
    const state = this.linearState(layerIndex);
    const nextConvState = detachLinearState(convState);
    const nextRecurrentState = detachLinearState(recurrentState);
    disposeLinearState(state);
    state.convState = nextConvState;
    state.recurrentState = nextRecurrentState;
  }

  [Symbol.dispose](): void {
    for (const layer of this.#layers) {
      if (layer.type === "full_attention") {
        layer.cache[Symbol.dispose]();
      } else {
        disposeLinearState(layer.state);
      }
    }
  }
}
