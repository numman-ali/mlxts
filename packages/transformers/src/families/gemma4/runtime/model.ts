/**
 * Gemma 4 model runtime helpers.
 * @module
 */

import {
  add,
  compile,
  type DisposableTransform,
  divide,
  type MxArray,
  multiply,
  reshape,
  retainArray,
  slice,
  tanh,
} from "@mlxts/core";
import type { Embedding, Linear } from "@mlxts/nn";
import { type AttentionMask, createStepAttentionMask } from "../../../infrastructure/masks";
import type { Gemma4RMSNorm } from "../norm";
import type { Gemma4LayerType, Gemma4SharedKeyValues } from "../types";

export interface SharedKeyValuePlan {
  sourceIndices: (number | null)[];
  retainedSourceFlags: boolean[];
}

export interface PerLayerInputModules {
  embedTokensPerLayer: Embedding;
  perLayerModelProjection: Linear;
  perLayerProjectionNorm: Gemma4RMSNorm;
}

export interface PerLayerInputConfig {
  layerCount: number;
  hiddenSizePerLayerInput: number;
  embeddingScale: number;
  inputScale: number;
  projectionScale: number;
}

export interface AttentionMaskLayer {
  selfAttention: {
    layerType: Gemma4LayerType;
  };
}

export function buildSharedKeyValuePlan(
  layerTypes: readonly Gemma4LayerType[],
  numKvSharedLayers: number,
): SharedKeyValuePlan {
  const sourceIndices = Array<number | null>(layerTypes.length).fill(null);
  const retainedSourceFlags = Array<boolean>(layerTypes.length).fill(false);

  if (numKvSharedLayers === 0) {
    return { sourceIndices, retainedSourceFlags };
  }

  const firstSharedLayerIndex = layerTypes.length - numKvSharedLayers;
  const latestSourceByType = new Map<Gemma4LayerType, number>();
  for (let layerIndex = 0; layerIndex < firstSharedLayerIndex; layerIndex += 1) {
    const layerType = layerTypes[layerIndex];
    if (layerType !== undefined) {
      latestSourceByType.set(layerType, layerIndex);
    }
  }

  for (let layerIndex = firstSharedLayerIndex; layerIndex < layerTypes.length; layerIndex += 1) {
    const layerType = layerTypes[layerIndex];
    const sourceIndex = layerType === undefined ? undefined : latestSourceByType.get(layerType);
    if (sourceIndex === undefined) {
      throw new Error(
        `Gemma4TextModel: shared KV layer ${layerIndex} has no earlier non-shared ${layerType ?? "unknown"} source.`,
      );
    }
    sourceIndices[layerIndex] = sourceIndex;
    retainedSourceFlags[sourceIndex] = true;
  }

  return { sourceIndices, retainedSourceFlags };
}

export function releaseSharedKeyValues(values: (Gemma4SharedKeyValues | null)[]): void {
  for (const value of values) {
    value?.keys.free();
    value?.values.free();
  }
}

export function releasePerLayerInputs(value: MxArray | null): void {
  value?.free();
}

function assertPerLayerInputShape(
  perLayerInputs: MxArray,
  context: string,
): { batch: number; sequenceLength: number } {
  const [batch, sequenceLength] = perLayerInputs.shape;
  if (batch === undefined || sequenceLength === undefined) {
    throw new Error(`${context}: per-layer inputs are missing batch or sequence axes.`);
  }
  return { batch, sequenceLength };
}

export function createPerLayerInputs(
  inputIds: MxArray,
  inputsEmbeds: MxArray,
  modules: PerLayerInputModules | null,
  config: PerLayerInputConfig,
): MxArray | null {
  if (modules === null || config.hiddenSizePerLayerInput === 0) {
    return null;
  }

  const [batch, sequenceLength] = inputIds.shape;
  if (batch === undefined || sequenceLength === undefined) {
    throw new Error("Gemma4TextModel.createPerLayerInputs: expected rank-2 token ids.");
  }

  using embeddedPerLayer = modules.embedTokensPerLayer.forward(inputIds);
  using scaledEmbeddedPerLayer = multiply(embeddedPerLayer, config.embeddingScale);
  using reshapedEmbeddedPerLayer = reshape(scaledEmbeddedPerLayer, [
    batch,
    sequenceLength,
    config.layerCount,
    config.hiddenSizePerLayerInput,
  ]);
  using projectedPerLayer = modules.perLayerModelProjection.forward(inputsEmbeds);
  using scaledProjectedPerLayer = multiply(projectedPerLayer, config.projectionScale);
  using reshapedProjectedPerLayer = reshape(scaledProjectedPerLayer, [
    batch,
    sequenceLength,
    config.layerCount,
    config.hiddenSizePerLayerInput,
  ]);
  using normalizedProjectedPerLayer =
    modules.perLayerProjectionNorm.forward(reshapedProjectedPerLayer);
  using combinedPerLayer = add(reshapedEmbeddedPerLayer, normalizedProjectedPerLayer);
  using scaledCombinedPerLayer = multiply(combinedPerLayer, config.inputScale);
  return retainArray(scaledCombinedPerLayer);
}

export function resolvePerLayerInput(
  perLayerInputs: MxArray | null,
  layerIndex: number,
  hiddenSizePerLayerInput: number,
  context: string,
): MxArray | null {
  if (perLayerInputs === null) {
    return null;
  }

  const { batch, sequenceLength } = assertPerLayerInputShape(perLayerInputs, context);
  using layerView = slice(
    perLayerInputs,
    [0, 0, layerIndex, 0],
    [batch, sequenceLength, layerIndex + 1, hiddenSizePerLayerInput],
  );
  return reshape(layerView, [batch, sequenceLength, hiddenSizePerLayerInput]);
}

export function takeSharedKeyValuesForLayer(
  sourceIndices: readonly (number | null)[],
  keyValues: readonly (Gemma4SharedKeyValues | null)[],
  layerIndex: number,
): Gemma4SharedKeyValues | undefined {
  const sourceIndex = sourceIndices[layerIndex];
  if (sourceIndex === null || sourceIndex === undefined) {
    return undefined;
  }

  const sharedKeyValues = keyValues[sourceIndex] ?? undefined;
  if (sharedKeyValues === undefined) {
    throw new Error(
      `Gemma4TextModel.forward: shared KV source ${sourceIndex} for layer ${layerIndex} is unavailable.`,
    );
  }
  return sharedKeyValues;
}

export function storeLayerKeyValues(
  retainedSourceFlags: readonly boolean[],
  keyValues: (Gemma4SharedKeyValues | null)[],
  layerIndex: number,
  nextKeyValues: Gemma4SharedKeyValues | null,
): void {
  if (nextKeyValues !== null && !retainedSourceFlags[layerIndex]) {
    nextKeyValues.keys.free();
    nextKeyValues.values.free();
    keyValues[layerIndex] = null;
    return;
  }
  keyValues[layerIndex] = nextKeyValues;
}

export function createAttentionMasks(
  layers: readonly AttentionMaskLayer[],
  sequenceLength: number,
  pastLength: number,
  slidingWindow: number,
  trimmedSlidingMasks: boolean,
): AttentionMask[] {
  const sharedMasks = new Map<string, AttentionMask>();
  return layers.map((layer) => {
    const windowSize =
      layer.selfAttention.layerType === "sliding_attention" ? slidingWindow : undefined;
    const key = windowSize === undefined ? "full" : `sliding:${windowSize}`;
    const existingMask = sharedMasks.get(key);
    if (existingMask !== undefined) {
      return existingMask;
    }
    const attentionMask = createStepAttentionMask(
      sequenceLength,
      pastLength,
      windowSize,
      windowSize !== undefined && trimmedSlidingMasks,
    );
    sharedMasks.set(key, attentionMask);
    return attentionMask;
  });
}

export function createLogitSoftcap(
  finalLogitSoftcapping: number | null,
): DisposableTransform<(logits: MxArray) => MxArray> | null {
  if (finalLogitSoftcapping === null) {
    return null;
  }

  return compile(
    (logits: MxArray) => {
      using scaledLogits = divide(logits, finalLogitSoftcapping);
      using tanhLogits = tanh(scaledLogits);
      return multiply(tanhLogits, finalLogitSoftcapping);
    },
    { shapeless: true },
  );
}
