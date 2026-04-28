/**
 * Shared Mixture-of-Experts routing and packed expert execution.
 * @module
 */

import {
  argpartition,
  broadcastTo,
  divide,
  expandDims,
  formatShape,
  type MxArray,
  matmul,
  multiply,
  reshape,
  slice,
  softmax,
  sum,
  takeAlongAxis,
  takeAxis,
  transpose,
  zeros,
} from "@mlxts/core";
import { Module } from "@mlxts/nn";

/** Top-k expert indices and normalized per-token weights. */
export type RoutedTopK = {
  indices: MxArray;
  weights: MxArray;
};

/** Activation used after packed gate/up expert projection. */
export type ExpertActivation = (gate: MxArray, value: MxArray) => MxArray;

function validateRouterProbabilities(
  probabilities: MxArray,
  topK: number,
  numExperts: number,
  context: string,
): { tokenCount: number } {
  const [tokenCount, expertCount] = probabilities.shape;
  if (probabilities.shape.length !== 2 || tokenCount === undefined || expertCount !== numExperts) {
    throw new Error(
      `${context}: expected router probabilities with shape [tokens, ${numExperts}], got ${formatShape(probabilities.shape)}.`,
    );
  }
  if (!Number.isInteger(topK) || topK <= 0 || topK > numExperts) {
    throw new Error(`${context}: topK must be in [1, ${numExperts}], got ${topK}.`);
  }
  return { tokenCount };
}

/** Select and normalize the top-k expert probabilities per token. */
export function topKFromRouterProbabilities(
  probabilities: MxArray,
  topK: number,
  numExperts: number,
  context = "topKFromRouterProbabilities",
): RoutedTopK {
  const { tokenCount } = validateRouterProbabilities(probabilities, topK, numExperts, context);
  using partitioned = argpartition(probabilities, numExperts - topK, -1);
  const indices = slice(partitioned, [0, numExperts - topK], [tokenCount, numExperts]);
  try {
    using weights = takeAlongAxis(probabilities, indices, -1);
    using denominator = sum(weights, -1, true);
    const normalizedWeights = divide(weights, denominator);
    return { indices, weights: normalizedWeights };
  } catch (error) {
    indices.free();
    throw error;
  }
}

/** Route logits through softmax and select normalized top-k expert weights. */
export function topKFromRouterLogits(
  logits: MxArray,
  topK: number,
  numExperts: number,
  context = "topKFromRouterLogits",
  options: { precise?: boolean } = {},
): RoutedTopK {
  using probabilities = softmax(logits, -1, { precise: options.precise ?? false });
  return topKFromRouterProbabilities(probabilities, topK, numExperts, context);
}

/** Packed SwitchGLU expert bank using `[experts, 2 * intermediate, hidden]` weights. */
export class PackedSwitchGLUExperts extends Module {
  gateUpProjection: MxArray;
  downProjection: MxArray;
  #activation: ExpertActivation;
  #hiddenSize: number;
  #intermediateSize: number;

  constructor(
    numExperts: number,
    hiddenSize: number,
    intermediateSize: number,
    activation: ExpertActivation,
  ) {
    super();
    this.#activation = activation;
    this.#hiddenSize = hiddenSize;
    this.#intermediateSize = intermediateSize;
    this.gateUpProjection = zeros([numExperts, 2 * intermediateSize, hiddenSize]);
    this.downProjection = zeros([numExperts, hiddenSize, intermediateSize]);
  }

  forward(hiddenStates: MxArray, topKIndices: MxArray, topKWeights: MxArray): MxArray {
    const [tokenCount, hiddenSize] = hiddenStates.shape;
    const [indexTokenCount, topK] = topKIndices.shape;
    if (
      hiddenStates.shape.length !== 2 ||
      tokenCount === undefined ||
      hiddenSize !== this.#hiddenSize
    ) {
      throw new Error(
        `PackedSwitchGLUExperts.forward: expected hidden states [tokens, ${this.#hiddenSize}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }
    if (topKIndices.shape.length !== 2 || indexTokenCount !== tokenCount || topK === undefined) {
      throw new Error(
        `PackedSwitchGLUExperts.forward: expected top-k indices [${tokenCount}, top_k], got ${formatShape(topKIndices.shape)}.`,
      );
    }
    if (topKWeights.shape[0] !== tokenCount || topKWeights.shape[1] !== topK) {
      throw new Error(
        `PackedSwitchGLUExperts.forward: top-k weights shape ${formatShape(topKWeights.shape)} must match indices ${formatShape(topKIndices.shape)}.`,
      );
    }

    using selectedGateUp = takeAxis(this.gateUpProjection, topKIndices, 0);
    using selectedDown = takeAxis(this.downProjection, topKIndices, 0);
    using hiddenWithTopK = expandDims(hiddenStates, 1);
    using hiddenForMatmul = expandDims(hiddenWithTopK, 2);
    using broadcastHidden = broadcastTo(hiddenForMatmul, [tokenCount, topK, 1, this.#hiddenSize]);
    using gateUpWeight = transpose(selectedGateUp, [0, 1, 3, 2]);
    using gateUpProjected = matmul(broadcastHidden, gateUpWeight);
    using gateUpFlat = reshape(gateUpProjected, [tokenCount, topK, 2 * this.#intermediateSize]);
    using gate = slice(gateUpFlat, [0, 0, 0], [tokenCount, topK, this.#intermediateSize]);
    using value = slice(
      gateUpFlat,
      [0, 0, this.#intermediateSize],
      [tokenCount, topK, 2 * this.#intermediateSize],
    );
    using activated = this.#activation(gate, value);
    using activatedForMatmul = expandDims(activated, 2);
    using downWeight = transpose(selectedDown, [0, 1, 3, 2]);
    using projected = matmul(activatedForMatmul, downWeight);
    using projectedFlat = reshape(projected, [tokenCount, topK, this.#hiddenSize]);
    using weights = expandDims(topKWeights, 2);
    using weighted = multiply(projectedFlat, weights);
    return sum(weighted, 1);
  }
}
