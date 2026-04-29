/**
 * Batch-aware hybrid cache for Qwen 3.5 / 3.6 text generation.
 * @module
 */

import {
  arange,
  array,
  concatenate,
  type DType,
  expandDims,
  greaterEqual,
  type MxArray,
  retainArray,
  slice,
  takeAxis,
} from "@mlxts/core";

import {
  BatchKVCache,
  type CacheLayerKind,
  cacheLayerKindsFromAttentionTypes,
} from "../../../infrastructure/cache";
import { disposeLayerStateSnapshot } from "../../../infrastructure/cache/runtime";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "../../../infrastructure/cache/view";
import type { TransformerBatchCache, TransformerCache } from "../../../types";
import type { Qwen3_5LayerType } from "../types";
import { type Qwen3_5LinearLayerState, Qwen3_5TextCache } from "./index";

function validateLayerTypes(layerTypes: readonly Qwen3_5LayerType[]): Qwen3_5LayerType[] {
  if (layerTypes.length === 0) {
    throw new Error("Qwen3_5TextBatchCache: layerTypes must contain at least one layer.");
  }
  return [...layerTypes];
}

function validateBatchMetadata(leftPadding: readonly number[]): number[] {
  if (leftPadding.length === 0) {
    throw new Error("Qwen3_5TextBatchCache: leftPadding must contain at least one request.");
  }
  return leftPadding.map((padding, index) => {
    if (!Number.isInteger(padding) || padding < 0) {
      throw new Error(
        `Qwen3_5TextBatchCache: leftPadding[${index}] must be a non-negative integer, got ${padding}.`,
      );
    }
    return padding;
  });
}

function validateBatchIndices(indices: readonly number[], batchSize: number): number[] {
  if (indices.length === 0) {
    throw new Error("Qwen3_5TextBatchCache.filter: batchIndices must contain at least one index.");
  }
  const seen = new Set<number>();
  return indices.map((index) => {
    if (!Number.isInteger(index) || index < 0 || index >= batchSize) {
      throw new Error(
        `Qwen3_5TextBatchCache.filter: batch index ${index} is out of range for batch size ${batchSize}.`,
      );
    }
    if (seen.has(index)) {
      throw new Error(`Qwen3_5TextBatchCache.filter: duplicate batch index ${index}.`);
    }
    seen.add(index);
    return index;
  });
}

function emptyLinearState(): Qwen3_5LinearLayerState {
  return {
    convState: null,
    recurrentState: null,
  };
}

function disposeLinearState(state: Qwen3_5LinearLayerState): void {
  state.convState?.free();
  state.recurrentState?.free();
  state.convState = null;
  state.recurrentState = null;
}

function retainOptionalArray(value: MxArray | null): MxArray | null {
  return value === null ? null : retainArray(value);
}

function filterOptionalArray(value: MxArray | null, indices: readonly number[]): MxArray | null {
  if (value === null) {
    return null;
  }
  using indexTensor = array([...indices], "int32");
  return takeAxis(value, indexTensor, 0);
}

function sliceBatch(value: MxArray | null, batchIndex: number): MxArray | null {
  if (value === null) {
    return null;
  }
  const stop = [...value.shape];
  stop[0] = batchIndex + 1;
  return slice(value, [batchIndex, ...Array(value.shape.length - 1).fill(0)], stop);
}

type LinearStateTensorSpec = {
  trailingShape: number[];
  dtype: DType;
};

function linearStateTensorSpec(value: MxArray): LinearStateTensorSpec {
  const [batchSize, ...trailingShape] = value.shape;
  if (batchSize === undefined || trailingShape.some((dimension) => dimension === undefined)) {
    throw new Error("Qwen3_5TextBatchCache.extend: expected concrete linear state shapes.");
  }
  return { trailingShape, dtype: value.dtype };
}

function compatibleLinearStateSpec(
  left: MxArray | null,
  right: MxArray | null,
  name: string,
): LinearStateTensorSpec | null {
  if (left === null && right === null) {
    return null;
  }
  if (left === null || right === null) {
    throw new Error(`Qwen3_5TextBatchCache.extend: ${name} must be initialized on both caches.`);
  }
  const spec = linearStateTensorSpec(left);
  const rightSpec = linearStateTensorSpec(right);
  if (
    spec.dtype !== rightSpec.dtype ||
    spec.trailingShape.length !== rightSpec.trailingShape.length ||
    spec.trailingShape.some((dimension, index) => dimension !== rightSpec.trailingShape[index])
  ) {
    throw new Error(`Qwen3_5TextBatchCache.extend: ${name} shapes and dtypes must match.`);
  }
  return spec;
}

function extendOptionalLinearArray(
  left: MxArray | null,
  right: MxArray | null,
  name: string,
): MxArray | null {
  const spec = compatibleLinearStateSpec(left, right, name);
  if (spec === null) {
    return null;
  }
  if (left === null || right === null) {
    throw new Error(`Qwen3_5TextBatchCache.extend: ${name} must be initialized on both caches.`);
  }
  using leftTensor = retainArray(left);
  using rightTensor = retainArray(right);
  return concatenate([leftTensor, rightTensor], 0);
}

function disposePreparedLinearStates(states: readonly Qwen3_5LinearLayerState[]): void {
  for (const state of states) {
    disposeLinearState(state);
  }
}

/** Batch cache for Qwen's mixed full-attention and linear-attention text layers. */
export class Qwen3_5TextBatchCache implements TransformerBatchCache {
  readonly #layerTypes: Qwen3_5LayerType[];
  readonly #layerKinds: readonly CacheLayerKind[];
  readonly #fullAttentionCache: BatchKVCache;
  readonly #linearStates: Qwen3_5LinearLayerState[];
  #linearLeftPadding: number[];

  constructor(layerTypes: readonly Qwen3_5LayerType[], leftPadding: readonly number[]) {
    this.#layerTypes = validateLayerTypes(layerTypes);
    this.#layerKinds = cacheLayerKindsFromAttentionTypes(this.#layerTypes, "Qwen3_5TextBatchCache");
    const padding = validateBatchMetadata(leftPadding);
    this.#fullAttentionCache = new BatchKVCache(this.#layerTypes.length, padding);
    this.#linearLeftPadding = [...padding];
    this.#linearStates = this.#layerTypes.map(() => emptyLinearState());
  }

  get layerCount(): number {
    return this.#layerTypes.length;
  }

  get layerKinds(): readonly CacheLayerKind[] {
    return this.#layerKinds;
  }

  get batchSize(): number {
    return this.#fullAttentionCache.batchSize;
  }

  get length(): number {
    return this.#fullAttentionCache.length;
  }

  get leftPadding(): readonly number[] {
    return this.#fullAttentionCache.leftPadding;
  }

  get offsets(): readonly number[] {
    return this.#fullAttentionCache.offsets;
  }

  isEmpty(): boolean {
    return (
      this.#fullAttentionCache.isEmpty() &&
      this.#linearStates.every((state) => state.convState === null && state.recurrentState === null)
    );
  }

  isTrimmable(): boolean {
    return false;
  }

  advance(sequenceLength: number): void {
    if (!Number.isInteger(sequenceLength) || sequenceLength < 0) {
      throw new Error(
        `Qwen3_5TextBatchCache.advance: sequenceLength must be a non-negative integer, got ${sequenceLength}.`,
      );
    }
    this.#fullAttentionCache.advance(sequenceLength);
    this.#linearLeftPadding = this.#linearLeftPadding.map((padding) =>
      Math.max(0, padding - sequenceLength),
    );
  }

  filter(batchIndices: readonly number[]): void {
    const indices = validateBatchIndices(batchIndices, this.batchSize);
    this.#fullAttentionCache.filter(indices);
    this.#linearLeftPadding = indices.map((index) => this.#linearLeftPadding[index] ?? 0);
    for (const state of this.#linearStates) {
      const nextConvState = filterOptionalArray(state.convState, indices);
      const nextRecurrentState = filterOptionalArray(state.recurrentState, indices);
      disposeLinearState(state);
      state.convState = nextConvState;
      state.recurrentState = nextRecurrentState;
    }
  }

  extend(other: TransformerBatchCache): void {
    if (!(other instanceof Qwen3_5TextBatchCache)) {
      throw new Error("Qwen3_5TextBatchCache.extend: expected another Qwen batch cache.");
    }
    if (other.layerCount !== this.layerCount) {
      throw new Error("Qwen3_5TextBatchCache.extend: layer counts must match.");
    }
    if (!this.hasSameLayerTypes(other)) {
      throw new Error("Qwen3_5TextBatchCache.extend: layer types must match.");
    }
    this.assertDecodeReadyForExtension(other);

    const nextLinearStates = this.prepareExtendedLinearStates(other);
    try {
      this.#fullAttentionCache.extend(other.#fullAttentionCache);
    } catch (error) {
      disposePreparedLinearStates(nextLinearStates);
      throw error;
    }

    for (let index = 0; index < this.#linearStates.length; index += 1) {
      const state = this.#linearStates[index];
      const nextState = nextLinearStates[index];
      if (state === undefined || nextState === undefined) {
        continue;
      }
      disposeLinearState(state);
      state.convState = nextState.convState;
      state.recurrentState = nextState.recurrentState;
      nextState.convState = null;
      nextState.recurrentState = null;
    }
    this.#linearLeftPadding = [...this.#linearLeftPadding, ...other.#linearLeftPadding];
  }

  restoreFromCache(batchIndex: number, source: TransformerCache): void {
    if (!(source instanceof Qwen3_5TextCache)) {
      throw new Error("Qwen3_5TextBatchCache.restoreFromCache: expected a Qwen text cache source.");
    }
    if (this.batchSize !== 1 || batchIndex !== 0) {
      throw new Error(
        "Qwen3_5TextBatchCache.restoreFromCache: seeded restore requires one batch row.",
      );
    }
    if (
      source.layerCount !== this.layerCount ||
      source.layerKinds.some((kind, index) => kind !== this.#layerKinds[index])
    ) {
      throw new Error("Qwen3_5TextBatchCache.restoreFromCache: layer patterns must match.");
    }

    for (let layerIndex = 0; layerIndex < this.#layerTypes.length; layerIndex += 1) {
      if (this.#layerTypes[layerIndex] === "full_attention") {
        const snapshot = source.cloneFullAttentionLayerState(layerIndex);
        try {
          this.#fullAttentionCache.restoreLayerState(layerIndex, snapshot);
        } finally {
          disposeLayerStateSnapshot(snapshot);
        }
        continue;
      }
      const state = source.linearState(layerIndex);
      this.updateLinearState(layerIndex, state.convState, state.recurrentState);
    }

    this.#fullAttentionCache.advance(source.offset);
    this.#linearLeftPadding = [0];
  }

  extract(batchIndex: number): TransformerCache {
    if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= this.batchSize) {
      throw new Error(
        `Qwen3_5TextBatchCache.extract: batch index ${batchIndex} is out of range for batch size ${this.batchSize}.`,
      );
    }
    const cache = new Qwen3_5TextCache(this.#layerTypes);
    for (let layerIndex = 0; layerIndex < this.#layerTypes.length; layerIndex += 1) {
      if (this.#layerTypes[layerIndex] === "full_attention") {
        this.extractFullAttentionLayer(cache, batchIndex, layerIndex);
      } else {
        this.extractLinearAttentionLayer(cache, batchIndex, layerIndex);
      }
    }
    cache.advance(Math.max(0, this.offsets[batchIndex] ?? 0));
    return cache;
  }

  offsetTensor(): MxArray {
    return this.#fullAttentionCache.offsetTensor();
  }

  leftPaddingTensor(): MxArray {
    return this.#fullAttentionCache.leftPaddingTensor();
  }

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    this.assertFullAttentionLayer(layerIndex, "updateAndFetch");
    return this.#fullAttentionCache.updateAndFetch(layerIndex, keys, values);
  }

  [INTERNAL_CACHE_VIEW](layerIndex: number, keys: MxArray, values: MxArray): TransformerCacheView {
    this.assertFullAttentionLayer(layerIndex, "cacheView");
    return this.#fullAttentionCache[INTERNAL_CACHE_VIEW](layerIndex, keys, values);
  }

  arrays(): MxArray[] {
    const arrays = this.#fullAttentionCache.arrays();
    for (const state of this.#linearStates) {
      if (state.convState !== null) {
        arrays.push(retainArray(state.convState));
      }
      if (state.recurrentState !== null) {
        arrays.push(retainArray(state.recurrentState));
      }
    }
    return arrays;
  }

  linearState(layerIndex: number): Qwen3_5LinearLayerState {
    this.assertLinearAttentionLayer(layerIndex, "linearState");
    const state = this.#linearStates[layerIndex];
    if (state === undefined) {
      throw new Error(`Qwen3_5TextBatchCache.linearState: layer ${layerIndex} is out of range.`);
    }
    return state;
  }

  updateLinearState(
    layerIndex: number,
    convState: MxArray | null,
    recurrentState: MxArray | null,
  ): void {
    const state = this.linearState(layerIndex);
    const nextConvState = retainOptionalArray(convState);
    const nextRecurrentState = retainOptionalArray(recurrentState);
    disposeLinearState(state);
    state.convState = nextConvState;
    state.recurrentState = nextRecurrentState;
  }

  linearAttentionMask(sequenceLength: number): MxArray | null {
    if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) {
      throw new Error(
        `Qwen3_5TextBatchCache.linearAttentionMask: sequenceLength must be positive, got ${sequenceLength}.`,
      );
    }
    if (this.#linearLeftPadding.every((padding) => padding === 0)) {
      return null;
    }
    using positions = arange(0, sequenceLength, 1, "int32");
    using positionRow = expandDims(positions, 0);
    using leftPadding = array([...this.#linearLeftPadding], "int32");
    using paddingColumn = expandDims(leftPadding, 1);
    return greaterEqual(positionRow, paddingColumn);
  }

  [Symbol.dispose](): void {
    this.#fullAttentionCache[Symbol.dispose]();
    for (const state of this.#linearStates) {
      disposeLinearState(state);
    }
  }

  private extractFullAttentionLayer(
    cache: Qwen3_5TextCache,
    batchIndex: number,
    layerIndex: number,
  ): void {
    const pair = this.#fullAttentionCache.extractLayer(batchIndex, layerIndex);
    if (pair === null) {
      return;
    }
    using keys = pair.keys;
    using values = pair.values;
    const updated = cache.updateAndFetch(layerIndex, keys, values);
    updated.keys.free();
    updated.values.free();
  }

  private extractLinearAttentionLayer(
    cache: Qwen3_5TextCache,
    batchIndex: number,
    layerIndex: number,
  ): void {
    const state = this.linearState(layerIndex);
    const convState = sliceBatch(state.convState, batchIndex);
    const recurrentState = sliceBatch(state.recurrentState, batchIndex);
    try {
      cache.updateLinearState(layerIndex, convState, recurrentState);
    } finally {
      convState?.free();
      recurrentState?.free();
    }
  }

  private assertFullAttentionLayer(layerIndex: number, operation: string): void {
    const layerType = this.#layerTypes[layerIndex];
    if (layerType !== "full_attention") {
      throw new Error(
        `Qwen3_5TextBatchCache.${operation}: layer ${layerIndex} is ${String(layerType)}; KV updates only apply to full_attention layers.`,
      );
    }
  }

  private assertLinearAttentionLayer(layerIndex: number, operation: string): void {
    const layerType = this.#layerTypes[layerIndex];
    if (layerType !== "linear_attention") {
      throw new Error(
        `Qwen3_5TextBatchCache.${operation}: layer ${layerIndex} is ${String(layerType)}; linear state only exists on linear_attention layers.`,
      );
    }
  }

  private hasSameLayerTypes(other: Qwen3_5TextBatchCache): boolean {
    return this.#layerTypes.every((layerType, index) => layerType === other.#layerTypes[index]);
  }

  private prepareExtendedLinearStates(other: Qwen3_5TextBatchCache): Qwen3_5LinearLayerState[] {
    const prepared: Qwen3_5LinearLayerState[] = [];
    try {
      for (let index = 0; index < this.#linearStates.length; index += 1) {
        const left = this.#linearStates[index];
        const right = other.#linearStates[index];
        if (left === undefined || right === undefined) {
          throw new Error("Qwen3_5TextBatchCache.extend: missing linear state.");
        }
        prepared.push({
          convState: extendOptionalLinearArray(
            left.convState,
            right.convState,
            `layer ${index} convState`,
          ),
          recurrentState: extendOptionalLinearArray(
            left.recurrentState,
            right.recurrentState,
            `layer ${index} recurrentState`,
          ),
        });
      }
      return prepared;
    } catch (error) {
      disposePreparedLinearStates(prepared);
      throw error;
    }
  }

  private assertDecodeReadyForExtension(other: Qwen3_5TextBatchCache): void {
    if (!this.#linearLeftPadding.every((padding) => padding === 0)) {
      throw new Error("Qwen3_5TextBatchCache.extend: target linear padding must be exhausted.");
    }
    if (!other.#linearLeftPadding.every((padding) => padding === 0)) {
      throw new Error("Qwen3_5TextBatchCache.extend: source linear padding must be exhausted.");
    }
  }
}
