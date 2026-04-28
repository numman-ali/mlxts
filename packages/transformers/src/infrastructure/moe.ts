/**
 * Shared Mixture-of-Experts routing and packed expert execution.
 * @module
 */

import type { QuantizationMode } from "@mlxts/core";
import {
  argpartition,
  broadcastTo,
  divide,
  expandDims,
  formatShape,
  gatherMm,
  gatherQmm,
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

/** Quantization parameters for split switch-linear checkpoint leaves. */
export type SwitchLinearQuantization = {
  groupSize: number;
  bits: number;
  mode: QuantizationMode;
};

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

/** One switch-routed expert projection, optionally backed by MLX quantized weights. */
export class SwitchLinear extends Module {
  weight: MxArray;
  scales: MxArray | null;
  biases: MxArray | null;
  #numExperts: number;
  #inputSize: number;
  #outputSize: number;
  #quantization: SwitchLinearQuantization | null = null;

  constructor(numExperts: number, inputSize: number, outputSize: number) {
    super();
    this.#numExperts = numExperts;
    this.#inputSize = inputSize;
    this.#outputSize = outputSize;
    this.weight = zeros([numExperts, outputSize, inputSize]);
    this.scales = null;
    this.biases = null;
  }

  /** Prepare this projection to receive MLX-native quantized switch-linear leaves. */
  prepareQuantized(params: SwitchLinearQuantization): void {
    if (this.#inputSize % params.groupSize !== 0) {
      throw new Error(
        `SwitchLinear.prepareQuantized: input size ${this.#inputSize} is not divisible by groupSize ${params.groupSize}.`,
      );
    }

    const packedInputSize = (this.#inputSize * params.bits) / 32;
    if (!Number.isInteger(packedInputSize)) {
      throw new Error(
        `SwitchLinear.prepareQuantized: input size ${this.#inputSize} and ${params.bits}-bit weights do not pack into uint32 columns.`,
      );
    }

    const scaleColumns = this.#inputSize / params.groupSize;
    this.weight.free();
    this.scales?.free();
    this.biases?.free();
    this.weight = zeros([this.#numExperts, this.#outputSize, packedInputSize], "uint32");
    this.scales = zeros([this.#numExperts, this.#outputSize, scaleColumns]);
    this.biases =
      params.mode === "affine" ? zeros([this.#numExperts, this.#outputSize, scaleColumns]) : null;
    this.#quantization = params;
  }

  /** Apply selected expert weights using MLX's matrix-gather kernels. */
  forward(hiddenStates: MxArray, topKIndices: MxArray): MxArray {
    const [indexTokenCount, topK] = topKIndices.shape;
    const tokenCount = hiddenStates.shape[0];
    const lastDimension = hiddenStates.shape[hiddenStates.shape.length - 1];
    if (
      hiddenStates.shape.length < 2 ||
      tokenCount === undefined ||
      lastDimension !== this.#inputSize
    ) {
      throw new Error(
        `SwitchLinear.forward: expected hidden states with first dimension tokens and last dimension ${this.#inputSize}, got ${formatShape(hiddenStates.shape)}.`,
      );
    }
    if (topKIndices.shape.length !== 2 || indexTokenCount !== tokenCount || topK === undefined) {
      throw new Error(
        `SwitchLinear.forward: expected top-k indices [${tokenCount}, top_k], got ${formatShape(topKIndices.shape)}.`,
      );
    }

    const quantization = this.#quantization;
    if (quantization !== null) {
      const scales = this.scales;
      if (scales === null) {
        throw new Error("SwitchLinear.forward: quantized projection is missing scales.");
      }

      const biases = this.biases;
      return gatherQmm(hiddenStates, this.weight, scales, {
        ...(biases === null ? {} : { biases }),
        rhsIndices: topKIndices,
        transpose: true,
        ...quantization,
      });
    }

    using transposedWeight = transpose(this.weight, [0, 2, 1]);
    return gatherMm(hiddenStates, transposedWeight, { rhsIndices: topKIndices });
  }
}

/** SwitchGLU expert bank using separate switch-routed gate, up, and down projections. */
export class SwitchGLUExperts extends Module {
  gateProjection: SwitchLinear;
  upProjection: SwitchLinear;
  downProjection: SwitchLinear;
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
    this.gateProjection = new SwitchLinear(numExperts, hiddenSize, intermediateSize);
    this.upProjection = new SwitchLinear(numExperts, hiddenSize, intermediateSize);
    this.downProjection = new SwitchLinear(numExperts, intermediateSize, hiddenSize);
  }

  /** Route hidden states through selected SwitchGLU experts and combine top-k outputs. */
  forward(hiddenStates: MxArray, topKIndices: MxArray, topKWeights: MxArray): MxArray {
    const [tokenCount, hiddenSize] = hiddenStates.shape;
    const [indexTokenCount, topK] = topKIndices.shape;
    if (
      hiddenStates.shape.length !== 2 ||
      tokenCount === undefined ||
      hiddenSize !== this.#hiddenSize
    ) {
      throw new Error(
        `SwitchGLUExperts.forward: expected hidden states [tokens, ${this.#hiddenSize}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }
    if (topKIndices.shape.length !== 2 || indexTokenCount !== tokenCount || topK === undefined) {
      throw new Error(
        `SwitchGLUExperts.forward: expected top-k indices [${tokenCount}, top_k], got ${formatShape(topKIndices.shape)}.`,
      );
    }
    if (topKWeights.shape[0] !== tokenCount || topKWeights.shape[1] !== topK) {
      throw new Error(
        `SwitchGLUExperts.forward: top-k weights shape ${formatShape(topKWeights.shape)} must match indices ${formatShape(topKIndices.shape)}.`,
      );
    }

    using hiddenWithTopK = expandDims(hiddenStates, 1);
    using hiddenForExperts = expandDims(hiddenWithTopK, 2);
    using gateProjected = this.gateProjection.forward(hiddenForExperts, topKIndices);
    using valueProjected = this.upProjection.forward(hiddenForExperts, topKIndices);
    using gate = reshape(gateProjected, [tokenCount, topK, this.#intermediateSize]);
    using value = reshape(valueProjected, [tokenCount, topK, this.#intermediateSize]);
    using activated = this.#activation(gate, value);
    using activatedForExperts = expandDims(activated, 2);
    using projected = this.downProjection.forward(activatedForExperts, topKIndices);
    using projectedFlat = reshape(projected, [tokenCount, topK, this.#hiddenSize]);
    using weights = expandDims(topKWeights, 2);
    using weighted = multiply(projectedFlat, weights);
    return sum(weighted, 1);
  }
}

/** Packed SwitchGLU expert bank using `[experts, 2 * intermediate, hidden]` weights. */
export class PackedSwitchGLUExperts extends Module {
  gateUpProjection: MxArray;
  downProjection: MxArray;
  #activation: ExpertActivation;
  #numExperts: number;
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
    this.#numExperts = numExperts;
    this.#hiddenSize = hiddenSize;
    this.#intermediateSize = intermediateSize;
    this.gateUpProjection = zeros([numExperts, 2 * intermediateSize, hiddenSize]);
    this.downProjection = zeros([numExperts, hiddenSize, intermediateSize]);
  }

  /** Create a split SwitchGLU bank with the same dimensions and activation. */
  toSwitchGLUExperts(): SwitchGLUExperts {
    return new SwitchGLUExperts(
      this.#numExperts,
      this.#hiddenSize,
      this.#intermediateSize,
      this.#activation,
    );
  }

  /** Route hidden states through selected packed SwitchGLU experts and combine top-k outputs. */
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
