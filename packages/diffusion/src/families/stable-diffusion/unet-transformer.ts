/**
 * Stable Diffusion UNet spatial transformer modules.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  formatShape,
  geluApprox,
  multiply,
  reshape,
  scaledDotProductAttention,
  split,
  transpose,
} from "@mlxts/core";
import { Conv2d, GroupNorm, LayerNorm, Linear, Module } from "@mlxts/nn";

function assertImage4d(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, height, width, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-4 NHWC input, got ${formatShape(x.shape)}.`);
  }
  return [batch, height, width, channels];
}

function assertSequence3d(x: MxArray, owner: string): readonly [number, number, number] {
  const [batch, sequence, channels] = x.shape;
  if (
    x.shape.length !== 3 ||
    batch === undefined ||
    sequence === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-3 sequence input, got ${formatShape(x.shape)}.`);
  }
  return [batch, sequence, channels];
}

function checkedModule<T>(modules: readonly T[], index: number, owner: string): T {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

function splitProjectionPair(projected: MxArray): { values: MxArray; gates: MxArray } {
  const parts = split(projected, 2, -1);
  const values = parts[0];
  const gates = parts[1];
  if (values === undefined || gates === undefined) {
    for (const part of parts) {
      part.free();
    }
    throw new Error("StableDiffusionUNetFeedForward.forward: GEGLU projection split failed.");
  }
  return { values, gates };
}

/** Multi-head attention used inside Stable Diffusion spatial transformer blocks. */
export class StableDiffusionUNetAttention extends Module {
  queryProjection: Linear;
  keyProjection: Linear;
  valueProjection: Linear;
  outputProjection: Linear;
  #numHeads: number;
  #headDim: number;
  #modelDims: number;

  constructor(modelDims: number, numHeads: number, keyValueDims = modelDims) {
    super();
    if (!Number.isInteger(numHeads) || numHeads <= 0 || modelDims % numHeads !== 0) {
      throw new Error("StableDiffusionUNetAttention: numHeads must divide modelDims.");
    }
    this.queryProjection = new Linear(modelDims, modelDims, false);
    this.keyProjection = new Linear(keyValueDims, modelDims, false);
    this.valueProjection = new Linear(keyValueDims, modelDims, false);
    this.outputProjection = new Linear(modelDims, modelDims);
    this.#numHeads = numHeads;
    this.#headDim = modelDims / numHeads;
    this.#modelDims = modelDims;
  }

  /** Run unmasked attention over `[batch, sequence, channels]` states. */
  forward(hiddenStates: MxArray, keyValueStates: MxArray): MxArray {
    const [batch, sequenceLength] = assertSequence3d(
      hiddenStates,
      "StableDiffusionUNetAttention.forward",
    );
    const [, keyValueLength] = assertSequence3d(
      keyValueStates,
      "StableDiffusionUNetAttention.forward",
    );
    using rawQueries = this.queryProjection.forward(hiddenStates);
    using rawKeys = this.keyProjection.forward(keyValueStates);
    using rawValues = this.valueProjection.forward(keyValueStates);
    using queries = reshape(rawQueries, [batch, sequenceLength, this.#numHeads, this.#headDim]);
    using keys = reshape(rawKeys, [batch, keyValueLength, this.#numHeads, this.#headDim]);
    using values = reshape(rawValues, [batch, keyValueLength, this.#numHeads, this.#headDim]);
    using transposedQueries = transpose(queries, [0, 2, 1, 3]);
    using transposedKeys = transpose(keys, [0, 2, 1, 3]);
    using transposedValues = transpose(values, [0, 2, 1, 3]);
    using attentionOutput = scaledDotProductAttention(
      transposedQueries,
      transposedKeys,
      transposedValues,
      {
        scale: this.#headDim ** -0.5,
      },
    );
    using outputSequence = transpose(attentionOutput, [0, 2, 1, 3]);
    using mergedOutput = reshape(outputSequence, [batch, sequenceLength, this.#modelDims]);
    return this.outputProjection.forward(mergedOutput);
  }
}

/** Feed-forward GEGLU block used by Stable Diffusion BasicTransformerBlock. */
export class StableDiffusionUNetFeedForward extends Module {
  projectionIn: Linear;
  projectionOut: Linear;

  constructor(modelDims: number, hiddenDims = modelDims * 4) {
    super();
    this.projectionIn = new Linear(modelDims, hiddenDims * 2);
    this.projectionOut = new Linear(hiddenDims, modelDims);
  }

  /** Run the transformer feed-forward projection. */
  forward(x: MxArray): MxArray {
    using projected = this.projectionIn.forward(x);
    const splitPair = splitProjectionPair(projected);
    try {
      using activatedGates = geluApprox(splitPair.gates);
      using gated = multiply(splitPair.values, activatedGates);
      return this.projectionOut.forward(gated);
    } finally {
      splitPair.values.free();
      splitPair.gates.free();
    }
  }
}

/** Basic transformer block used inside Stable Diffusion spatial transformer layers. */
export class StableDiffusionUNetTransformerBlock extends Module {
  norm1: LayerNorm;
  attention1: StableDiffusionUNetAttention;
  norm2: LayerNorm;
  attention2: StableDiffusionUNetAttention;
  norm3: LayerNorm;
  feedForward: StableDiffusionUNetFeedForward;

  constructor(modelDims: number, numHeads: number, crossAttentionDims: number) {
    super();
    this.norm1 = new LayerNorm(modelDims);
    this.attention1 = new StableDiffusionUNetAttention(modelDims, numHeads);
    this.norm2 = new LayerNorm(modelDims);
    this.attention2 = new StableDiffusionUNetAttention(modelDims, numHeads, crossAttentionDims);
    this.norm3 = new LayerNorm(modelDims);
    this.feedForward = new StableDiffusionUNetFeedForward(modelDims);
  }

  /** Run self-attention, cross-attention, and feed-forward residual updates. */
  forward(hiddenStates: MxArray, encoderHiddenStates: MxArray): MxArray {
    using normalizedSelf = this.norm1.forward(hiddenStates);
    using selfAttention = this.attention1.forward(normalizedSelf, normalizedSelf);
    using selfResidual = add(hiddenStates, selfAttention);
    using normalizedCross = this.norm2.forward(selfResidual);
    using crossAttention = this.attention2.forward(normalizedCross, encoderHiddenStates);
    using crossResidual = add(selfResidual, crossAttention);
    using normalizedFeedForward = this.norm3.forward(crossResidual);
    using feedForward = this.feedForward.forward(normalizedFeedForward);
    return add(crossResidual, feedForward);
  }
}

/** Spatial transformer block used in Stable Diffusion cross-attention UNet stages. */
export class StableDiffusionUNetTransformer2d extends Module {
  norm: GroupNorm;
  projectionIn: Linear | Conv2d;
  transformerBlocks: StableDiffusionUNetTransformerBlock[];
  projectionOut: Linear | Conv2d;
  #channels: number;
  #useLinearProjection: boolean;

  constructor(options: {
    channels: number;
    crossAttentionDims: number;
    numHeads: number;
    layers: number;
    normGroups: number;
    useLinearProjection: boolean;
  }) {
    super();
    this.norm = new GroupNorm(options.normGroups, options.channels);
    this.projectionIn = options.useLinearProjection
      ? new Linear(options.channels, options.channels)
      : new Conv2d(options.channels, options.channels, 1, 1, 0);
    this.transformerBlocks = Array.from(
      { length: options.layers },
      () =>
        new StableDiffusionUNetTransformerBlock(
          options.channels,
          options.numHeads,
          options.crossAttentionDims,
        ),
    );
    this.projectionOut = options.useLinearProjection
      ? new Linear(options.channels, options.channels)
      : new Conv2d(options.channels, options.channels, 1, 1, 0);
    this.#channels = options.channels;
    this.#useLinearProjection = options.useLinearProjection;
  }

  /** Run spatial transformer layers over an NHWC feature map. */
  forward(x: MxArray, encoderHiddenStates: MxArray): MxArray {
    const [batch, height, width] = assertImage4d(x, "StableDiffusionUNetTransformer2d.forward");
    using normalized = this.norm.forward(x);
    let hidden = this.#projectInput(normalized, batch, height, width);
    try {
      for (let index = 0; index < this.transformerBlocks.length; index += 1) {
        const block = checkedModule(
          this.transformerBlocks,
          index,
          "StableDiffusionUNetTransformer2d.forward",
        );
        const nextHidden = block.forward(hidden, encoderHiddenStates);
        hidden.free();
        hidden = nextHidden;
      }
      using projected = this.#projectOutput(hidden, batch, height, width);
      return add(x, projected);
    } catch (error) {
      hidden.free();
      throw error;
    }
  }

  #projectInput(x: MxArray, batch: number, height: number, width: number): MxArray {
    if (this.#useLinearProjection) {
      using sequence = reshape(x, [batch, height * width, this.#channels]);
      return this.projectionIn.forward(sequence);
    }
    using projected = this.projectionIn.forward(x);
    return reshape(projected, [batch, height * width, this.#channels]);
  }

  #projectOutput(x: MxArray, batch: number, height: number, width: number): MxArray {
    if (this.#useLinearProjection) {
      using projected = this.projectionOut.forward(x);
      return reshape(projected, [batch, height, width, this.#channels]);
    }
    using image = reshape(x, [batch, height, width, this.#channels]);
    return this.projectionOut.forward(image);
  }
}
