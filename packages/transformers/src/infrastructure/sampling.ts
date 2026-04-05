/**
 * Sampling helpers over model logits.
 * @module
 */

import {
  arange,
  argmax,
  argpartition,
  argsort,
  array,
  broadcastTo,
  concatenate,
  cumsum,
  divide,
  exp,
  full,
  greater,
  less,
  logsumexp,
  MxArray,
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

import type { SamplerOptions } from "../types";

function inferLastAxisSize(logits: MxArray): number {
  const axisSize = logits.shape[logits.shape.length - 1];
  if (axisSize === undefined || logits.shape.length === 0) {
    throw new Error(
      `sampleTokenTensor: expected logits with a trailing vocabulary axis, got [${logits.shape.join(", ")}].`,
    );
  }
  return axisSize;
}

function applyRepetitionPenaltyTensor(
  logits: MxArray,
  historyIds: MxArray | null,
  penalty: number | undefined,
): MxArray {
  if (historyIds === null || penalty === undefined || penalty <= 1.0) {
    return retainArray(logits);
  }

  using selectedLogits = takeAlongAxis(logits, historyIds, -1);
  using negativeMask = less(selectedLogits, 0);
  using penalizedNegative = multiply(selectedLogits, penalty);
  using penalizedPositive = divide(selectedLogits, penalty);
  using adjustedLogits = where(negativeMask, penalizedNegative, penalizedPositive);
  return putAlongAxis(logits, historyIds, adjustedLogits, -1);
}

function shapeToArray(shape: readonly number[]): number[] {
  return [...shape];
}

function applyTopPTensor(logprobs: MxArray, topP: number | undefined): MxArray {
  if (topP === undefined || topP <= 0 || topP >= 1) {
    return retainArray(logprobs);
  }

  const vocabSize = inferLastAxisSize(logprobs);
  using probabilities = exp(logprobs);
  using sortedIndices = argsort(logprobs, -1);
  using sortedProbabilities = takeAlongAxis(probabilities, sortedIndices, -1);
  using cumulativeSortedProbabilities = cumsum(sortedProbabilities, -1);
  using order = arange(0, vocabSize, 1, sortedIndices.dtype);
  using reshapedOrder = reshape(order, [1, vocabSize]);
  using broadcastOrder = broadcastTo(reshapedOrder, shapeToArray(sortedIndices.shape));
  using inverseBase = zerosForIndices(shapeToArray(sortedIndices.shape), sortedIndices.dtype);
  using inverseIndices = putAlongAxis(inverseBase, sortedIndices, broadcastOrder, -1);
  using cumulativeProbabilities = takeAlongAxis(cumulativeSortedProbabilities, inverseIndices, -1);
  using keepMask = greater(cumulativeProbabilities, 1 - topP);
  return where(keepMask, logprobs, Number.NEGATIVE_INFINITY);
}

function applyMinPTensor(logprobs: MxArray, minP: number | undefined): MxArray {
  if (minP === undefined || minP <= 0) {
    return retainArray(logprobs);
  }

  using maxLogprobs = max(logprobs, -1, true);
  using threshold = addScalar(maxLogprobs, Math.log(minP));
  using removeMask = less(logprobs, threshold);
  return where(removeMask, Number.NEGATIVE_INFINITY, logprobs);
}

function applyTopKTensor(logprobs: MxArray, topK: number | undefined): MxArray {
  const vocabSize = inferLastAxisSize(logprobs);
  if (topK === undefined || topK <= 0 || topK >= vocabSize) {
    return retainArray(logprobs);
  }

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
}

function addScalar(value: MxArray, amount: number): MxArray {
  return subtract(value, -amount);
}

function zerosForIndices(shape: number[], dtype: MxArray["dtype"]): MxArray {
  return full(shape, 0, dtype);
}

function logprobsForSampling(
  logits: MxArray,
  historyIds: MxArray | null,
  options: SamplerOptions,
): MxArray {
  using repetitionAdjusted = applyRepetitionPenaltyTensor(
    logits,
    historyIds,
    options.repetitionPenalty,
  );
  using normalizer = logsumexp(repetitionAdjusted, -1, true);
  let filteredLogprobs = subtract(repetitionAdjusted, normalizer);

  try {
    using topPFiltered = applyTopPTensor(filteredLogprobs, options.topP);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(topPFiltered);

    using minPFiltered = applyMinPTensor(filteredLogprobs, options.minP);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(minPFiltered);

    using topKFiltered = applyTopKTensor(filteredLogprobs, options.topK);
    filteredLogprobs.free();
    filteredLogprobs = retainArray(topKFiltered);

    return filteredLogprobs;
  } catch (error) {
    filteredLogprobs.free();
    throw error;
  }
}

function usesProbabilityFiltering(options: SamplerOptions): boolean {
  return (options.topK ?? 0) > 0 || (options.topP ?? 1) < 1 || (options.minP ?? 0) > 0;
}

function sampleCategoricalToken(
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
  return reshape(sampled, [1, 1]);
}

/** Stateful GPU-side sampler data for generation loops. */
export class SamplerState implements Disposable {
  #rngKey: MxArray | null;
  #historyIds: MxArray | null;
  #tracksHistory: boolean;

  constructor(promptTokenIds: readonly number[], options: SamplerOptions) {
    this.#rngKey = options.seed === undefined ? null : random.key(options.seed);
    this.#tracksHistory = (options.repetitionPenalty ?? 1) > 1;
    this.#historyIds =
      this.#tracksHistory && promptTokenIds.length > 0 ? array([promptTokenIds], "int32") : null;
  }

  sampleTokenTensor(logits: MxArray, options: SamplerOptions = {}): MxArray {
    const temperature = options.temperature ?? 1.0;
    using repetitionAdjusted = applyRepetitionPenaltyTensor(
      logits,
      this.#historyIds,
      options.repetitionPenalty,
    );

    if (temperature <= 0) {
      using greedy = argmax(repetitionAdjusted, -1);
      return reshape(greedy, [1, 1]);
    }

    const rngKey = this.takeRngKey();

    try {
      if (!usesProbabilityFiltering(options)) {
        using normalizer = logsumexp(repetitionAdjusted, -1, true);
        using logprobs = subtract(repetitionAdjusted, normalizer);
        return sampleCategoricalToken(logprobs, temperature, rngKey);
      }

      using logprobs = logprobsForSampling(logits, this.#historyIds, options);
      return sampleCategoricalToken(logprobs, temperature, rngKey);
    } finally {
      rngKey?.free();
    }
  }

  appendToken(tokenId: number | MxArray): void {
    if (!this.#tracksHistory) {
      return;
    }

    const tokenTensor =
      tokenId instanceof MxArray ? retainArray(tokenId) : array([[tokenId]], "int32");

    try {
      if (this.#historyIds === null) {
        this.#historyIds = tokenTensor;
        return;
      }

      const nextHistory = concatenate([this.#historyIds, tokenTensor], 1);
      this.#historyIds.free();
      this.#historyIds = nextHistory;
    } finally {
      if (this.#historyIds !== tokenTensor) {
        tokenTensor.free();
      }
    }
  }

  [Symbol.dispose](): void {
    this.#rngKey?.free();
    this.#historyIds?.free();
    this.#rngKey = null;
    this.#historyIds = null;
  }

  private takeRngKey(): MxArray | null {
    if (this.#rngKey === null) {
      return null;
    }

    const [nextState, sampleKey] = random.split(this.#rngKey);
    this.#rngKey.free();
    this.#rngKey = nextState;
    return sampleKey;
  }
}
