/**
 * Internal sampling runtime helpers.
 * @module
 */

import {
  arange,
  argpartition,
  argsort,
  broadcastTo,
  compile,
  cumsum,
  type DisposableTransform,
  divide,
  exp,
  full,
  greater,
  less,
  logsumexp,
  type MxArray,
  max,
  multiply,
  negative,
  putAlongAxis,
  random,
  reshape,
  retainArray,
  subtract,
  takeAlongAxis,
  takeAxis,
  where,
} from "@mlxts/core";

import type { SamplerOptions } from "../../types";

const repetitionPenaltyVariantsByPenalty = new Map<
  number,
  DisposableTransform<(logits: MxArray, historyIds: MxArray) => MxArray>
>();
const minPVariantsByThreshold = new Map<
  number,
  DisposableTransform<(logprobs: MxArray) => MxArray>
>();
const topPVariantsByShape = new Map<string, DisposableTransform<(logprobs: MxArray) => MxArray>>();
const topKVariantsByShape = new Map<string, DisposableTransform<(logprobs: MxArray) => MxArray>>();

function repetitionPenaltyTransformFor(
  penalty: number,
): DisposableTransform<(logits: MxArray, historyIds: MxArray) => MxArray> {
  let transform = repetitionPenaltyVariantsByPenalty.get(penalty);
  if (transform !== undefined) {
    return transform;
  }

  transform = compile(
    (logits: MxArray, historyIds: MxArray) => {
      using selectedLogits = takeAlongAxis(logits, historyIds, -1);
      using negativeMask = less(selectedLogits, 0);
      using penalizedNegative = multiply(selectedLogits, penalty);
      using penalizedPositive = divide(selectedLogits, penalty);
      using adjustedLogits = where(negativeMask, penalizedNegative, penalizedPositive);
      return putAlongAxis(logits, historyIds, adjustedLogits, -1);
    },
    { shapeless: true },
  );
  repetitionPenaltyVariantsByPenalty.set(penalty, transform);
  return transform;
}

function minPTransformFor(minP: number): DisposableTransform<(logprobs: MxArray) => MxArray> {
  let transform = minPVariantsByThreshold.get(minP);
  if (transform !== undefined) {
    return transform;
  }

  transform = compile(
    (logprobs: MxArray) => {
      using maxLogprobs = max(logprobs, -1, true);
      using threshold = subtract(maxLogprobs, -Math.log(minP));
      using removeMask = less(logprobs, threshold);
      return where(removeMask, Number.NEGATIVE_INFINITY, logprobs);
    },
    { shapeless: true },
  );
  minPVariantsByThreshold.set(minP, transform);
  return transform;
}

function inferLastAxisSize(logits: MxArray): number {
  const axisSize = logits.shape[logits.shape.length - 1];
  if (axisSize === undefined || logits.shape.length === 0) {
    throw new Error(
      `sampleTokenTensor: expected logits with a trailing vocabulary axis, got [${logits.shape.join(", ")}].`,
    );
  }
  return axisSize;
}

function shapeToArray(shape: readonly number[]): number[] {
  return [...shape];
}

function shapeVariantKey(logprobs: MxArray, configValue: number): string {
  return `${configValue}|${logprobs.dtype}|${logprobs.shape.join("x")}`;
}

function zerosForIndices(shape: number[], dtype: MxArray["dtype"]): MxArray {
  return full(shape, 0, dtype);
}

function topPTransformFor(
  logprobs: MxArray,
  topP: number,
): DisposableTransform<(logprobs: MxArray) => MxArray> {
  const key = shapeVariantKey(logprobs, topP);
  let transform = topPVariantsByShape.get(key);
  if (transform !== undefined) {
    return transform;
  }

  const vocabSize = inferLastAxisSize(logprobs);
  const sortedShape = shapeToArray(logprobs.shape);
  transform = compile((logprobs: MxArray) => {
    using probabilities = exp(logprobs);
    using sortedIndices = argsort(logprobs, -1);
    using sortedProbabilities = takeAlongAxis(probabilities, sortedIndices, -1);
    using cumulativeSortedProbabilities = cumsum(sortedProbabilities, -1);
    using order = arange(0, vocabSize, 1, sortedIndices.dtype);
    using reshapedOrder = reshape(order, [1, vocabSize]);
    using broadcastOrder = broadcastTo(reshapedOrder, sortedShape);
    using inverseBase = zerosForIndices(sortedShape, sortedIndices.dtype);
    using inverseIndices = putAlongAxis(inverseBase, sortedIndices, broadcastOrder, -1);
    using cumulativeProbabilities = takeAlongAxis(
      cumulativeSortedProbabilities,
      inverseIndices,
      -1,
    );
    using keepMask = greater(cumulativeProbabilities, 1 - topP);
    return where(keepMask, logprobs, Number.NEGATIVE_INFINITY);
  });
  topPVariantsByShape.set(key, transform);
  return transform;
}

function topKTransformFor(
  logprobs: MxArray,
  topK: number,
): DisposableTransform<(logprobs: MxArray) => MxArray> {
  const key = shapeVariantKey(logprobs, topK);
  let transform = topKVariantsByShape.get(key);
  if (transform !== undefined) {
    return transform;
  }

  const vocabSize = inferLastAxisSize(logprobs);
  transform = compile((logprobs: MxArray) => {
    using negatedLogprobs = negative(logprobs);
    using partitionedIndices = argpartition(negatedLogprobs, topK - 1, -1);
    using maskedPositions = arange(topK, vocabSize, 1, "int32");
    using maskedIndices = takeAxis(partitionedIndices, maskedPositions, -1);
    using maskedValues = full(
      shapeToArray(maskedIndices.shape),
      Number.NEGATIVE_INFINITY,
      logprobs.dtype,
    );
    return putAlongAxis(logprobs, maskedIndices, maskedValues, -1);
  });
  topKVariantsByShape.set(key, transform);
  return transform;
}

function applyTopP(logprobs: MxArray, topP: number | undefined): MxArray {
  if (topP === undefined || topP <= 0 || topP >= 1) {
    return retainArray(logprobs);
  }

  return topPTransformFor(logprobs, topP)(logprobs);
}

function applyMinP(logprobs: MxArray, minP: number | undefined): MxArray {
  if (minP === undefined || minP <= 0) {
    return retainArray(logprobs);
  }

  return minPTransformFor(minP)(logprobs);
}

function applyTopK(logprobs: MxArray, topK: number | undefined): MxArray {
  const vocabSize = inferLastAxisSize(logprobs);
  if (topK === undefined || topK <= 0 || topK >= vocabSize) {
    return retainArray(logprobs);
  }

  return topKTransformFor(logprobs, topK)(logprobs);
}

export function applySamplingRepetitionPenalty(
  logits: MxArray,
  historyIds: MxArray | null,
  penalty: number | undefined,
): MxArray {
  if (historyIds === null || penalty === undefined || penalty <= 1.0) {
    return retainArray(logits);
  }

  return repetitionPenaltyTransformFor(penalty)(logits, historyIds);
}

export function normalizeSamplingLogits(logits: MxArray): MxArray {
  using normalizer = logsumexp(logits, -1, true);
  return subtract(logits, normalizer);
}

export function buildFilteredSamplingLogprobs(logits: MxArray, options: SamplerOptions): MxArray {
  let filteredLogprobs = normalizeSamplingLogits(logits);

  try {
    using topPFiltered = applyTopP(filteredLogprobs, options.topP);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(topPFiltered);

    using minPFiltered = applyMinP(filteredLogprobs, options.minP);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(minPFiltered);

    using topKFiltered = applyTopK(filteredLogprobs, options.topK);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(topKFiltered);

    return filteredLogprobs;
  } catch (error) {
    filteredLogprobs.free();
    throw error;
  }
}

export function usesProbabilityFiltering(options: SamplerOptions): boolean {
  return (options.topK ?? 0) > 0 || (options.topP ?? 1) < 1 || (options.minP ?? 0) > 0;
}

export function sampleCategoricalToken(
  logprobs: MxArray,
  temperature: number,
  rngKey: MxArray | null,
): MxArray {
  using scaledLogprobs =
    temperature === 1.0 ? retainArray(logprobs) : divide(logprobs, temperature);
  using sampled =
    rngKey === null
      ? random.categorical(scaledLogprobs, -1)
      : random.categorical(scaledLogprobs, -1, rngKey);
  const token = reshape(sampled, [1, 1]);
  // The categorical graph closes over RNG state; materialize before local key handles are freed.
  token.eval();
  return token;
}
