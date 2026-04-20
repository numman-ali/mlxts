/**
 * Gemma 4 attention runtime helpers.
 * @module
 */

import { type MxArray, reshape, scaledDotProductAttention, transpose } from "@mlxts/core";
import type { Linear } from "@mlxts/nn";

import type { AttentionMask } from "../../../infrastructure/masks";
import type { Gemma4RMSNorm } from "../norm";
import type { RotaryEmbedding } from "../rope";

export interface AttentionRuntimeLayout {
  numHeads: number;
  numKeyValueHeads: number;
  headDim: number;
}

export interface AttentionRuntimeWeights {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear | null;
  outputProjection: Linear;
  qNorm: Gemma4RMSNorm;
  kNorm: Gemma4RMSNorm;
  vNorm: Gemma4RMSNorm;
  rope: RotaryEmbedding;
}

function reshapeHeads(
  projected: MxArray,
  batchSize: number,
  sequenceLength: number,
  numHeads: number,
  headDim: number,
): MxArray {
  using inputs = reshape(projected, [batchSize, sequenceLength, numHeads, headDim]);
  return transpose(inputs, [0, 2, 1, 3]);
}

function prepareQueryHeads(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  x: MxArray,
): MxArray {
  const batchSize = x.shape[0] ?? 0;
  const sequenceLength = x.shape[1] ?? 0;
  using projectedQueries = weights.qProjection.forward(x);
  using queryHeads = reshapeHeads(
    projectedQueries,
    batchSize,
    sequenceLength,
    layout.numHeads,
    layout.headDim,
  );
  return weights.qNorm.forward(queryHeads);
}

function prepareKeyHeads(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  x: MxArray,
): MxArray {
  const batchSize = x.shape[0] ?? 0;
  const sequenceLength = x.shape[1] ?? 0;
  using projectedKeys = weights.kProjection.forward(x);
  using keyHeads = reshapeHeads(
    projectedKeys,
    batchSize,
    sequenceLength,
    layout.numKeyValueHeads,
    layout.headDim,
  );
  return weights.kNorm.forward(keyHeads);
}

export function prepareQueryHeadsAndRope(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  x: MxArray,
  offset: number,
): MxArray {
  using queryHeads = prepareQueryHeads(layout, weights, x);
  return weights.rope.forward(queryHeads, offset);
}

export function prepareKeyHeadsAndRope(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  x: MxArray,
  offset: number,
): MxArray {
  using keyHeads = prepareKeyHeads(layout, weights, x);
  return weights.rope.forward(keyHeads, offset);
}

export function prepareKeyValueHeads(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  x: MxArray,
  offset: number,
): { keys: MxArray; values: MxArray } {
  if (weights.vProjection === null) {
    const batchSize = x.shape[0] ?? 0;
    const sequenceLength = x.shape[1] ?? 0;
    using projectedKeys = weights.kProjection.forward(x);
    using keyInputs = reshapeHeads(
      projectedKeys,
      batchSize,
      sequenceLength,
      layout.numKeyValueHeads,
      layout.headDim,
    );
    using normalizedKeys = weights.kNorm.forward(keyInputs);
    const valueHeads = weights.vNorm.forward(keyInputs);
    let rotatedKeys: MxArray | null = null;

    try {
      rotatedKeys = weights.rope.forward(normalizedKeys, offset);
      return { keys: rotatedKeys, values: valueHeads };
    } catch (error) {
      valueHeads.free();
      rotatedKeys?.free();
      throw error;
    }
  }

  const keys = prepareKeyHeadsAndRope(layout, weights, x, offset);
  let values: MxArray | null = null;

  try {
    const batchSize = x.shape[0] ?? 0;
    const sequenceLength = x.shape[1] ?? 0;
    using projectedValues = weights.vProjection.forward(x);
    using valueInputs = reshapeHeads(
      projectedValues,
      batchSize,
      sequenceLength,
      layout.numKeyValueHeads,
      layout.headDim,
    );
    values = weights.vNorm.forward(valueInputs);
    return { keys, values };
  } catch (error) {
    keys.free();
    values?.free();
    throw error;
  }
}

function projectAttentionOutput(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  attentionOutput: MxArray,
): MxArray {
  const batchSize = attentionOutput.shape[0] ?? 0;
  const sequenceLength = attentionOutput.shape[2] ?? 0;
  using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
  using mergedOutput = reshape(transposedOutput, [
    batchSize,
    sequenceLength,
    layout.numHeads * layout.headDim,
  ]);
  return weights.outputProjection.forward(mergedOutput);
}

export function runSdpaAndOutput(
  layout: AttentionRuntimeLayout,
  weights: AttentionRuntimeWeights,
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  attentionMask: AttentionMask,
): MxArray {
  const attentionOptions =
    attentionMask === null
      ? { scale: 1.0 }
      : attentionMask === "causal"
        ? { scale: 1.0, maskMode: "causal" as const }
        : { scale: 1.0, maskArray: attentionMask };
  using attentionOutput = scaledDotProductAttention(queries, keys, values, attentionOptions);
  return projectAttentionOutput(layout, weights, attentionOutput);
}
