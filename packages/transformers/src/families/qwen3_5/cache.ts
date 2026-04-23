/**
 * Family-local hybrid cache for Qwen 3.5 text layers.
 * @module
 */

import { type MxArray, retainArray } from "@mlxts/core";

import { KVCache } from "../../infrastructure/cache";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "../../infrastructure/cache/view";
import type { TransformerCache } from "../../types";
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
