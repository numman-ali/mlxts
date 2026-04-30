/**
 * FLUX.1 transformer blocks.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  concatenate,
  divide,
  geluApprox,
  mean,
  multiply,
  reshape,
  retainArray,
  split,
  sqrt,
  square,
  subtract,
  transpose,
} from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

import { type FluxAttentionProjection, FluxSelfAttention, fluxAttention } from "./attention";
import { assertSequence3d, freeArrays, sliceAxis, sliceLastAxis } from "./tensor-utils";

type FluxModulationTriplet = {
  shift: MxArray;
  scale: MxArray;
  gate: MxArray;
};

type FluxModulationOutput =
  | { kind: "single"; attention: FluxModulationTriplet }
  | {
      kind: "double";
      attention: FluxModulationTriplet;
      feedForward: FluxModulationTriplet;
    };

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function disposeTriplet(triplet: FluxModulationTriplet): void {
  triplet.shift.free();
  triplet.scale.free();
  triplet.gate.free();
}

function disposeModulation(output: FluxModulationOutput): void {
  disposeTriplet(output.attention);
  if (output.kind === "double") {
    disposeTriplet(output.feedForward);
  }
}

function affineFreeLayerNorm(x: MxArray, dims: number, owner: string): MxArray {
  const shape = assertSequence3d(x, owner);
  if (shape.channels !== dims) {
    throw new Error(`${owner}: expected last dimension ${dims}, got ${shape.channels}.`);
  }
  using center = mean(x, -1, true);
  using centered = subtract(x, center);
  using squared = square(centered);
  using variance = mean(squared, -1, true);
  using stabilized = add(variance, 1e-6);
  using denominator = sqrt(stabilized);
  return divide(centered, denominator);
}

function applyModulation(x: MxArray, shift: MxArray, scale: MxArray): MxArray {
  using scaled = add(scale, 1);
  using multiplied = multiply(x, scaled);
  return add(multiplied, shift);
}

function reshapeModulation(part: MxArray, batch: number, hiddenSize: number): MxArray {
  return reshape(part, [batch, 1, hiddenSize]);
}

function disposeProjection(projection: FluxAttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
}

/** Gated modulation projection used by FLUX blocks. */
export class FluxModulation extends Module {
  linear: Linear;
  #hiddenSize: number;
  #double: boolean;

  constructor(hiddenSize: number, double: boolean) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * (double ? 6 : 3));
    this.#hiddenSize = hiddenSize;
    this.#double = double;
  }

  forward(vec: MxArray): MxArray {
    const [batch, channels] = vec.shape;
    if (vec.shape.length !== 2 || batch === undefined || channels !== this.#hiddenSize) {
      throw new Error("FluxModulation.forward: vec must have shape [batch, hiddenSize].");
    }
    using activated = silu(vec);
    return this.linear.forward(activated);
  }

  modulate(vec: MxArray): FluxModulationOutput {
    const [batch] = vec.shape;
    if (batch === undefined) {
      throw new Error("FluxModulation.modulate: vec must include a batch dimension.");
    }
    using projected = this.forward(vec);
    const partCount = this.#double ? 6 : 3;
    const parts = split(projected, partCount, -1);
    try {
      const attention = {
        shift: reshapeModulation(
          partAt(parts, 0, "FluxModulation.modulate"),
          batch,
          this.#hiddenSize,
        ),
        scale: reshapeModulation(
          partAt(parts, 1, "FluxModulation.modulate"),
          batch,
          this.#hiddenSize,
        ),
        gate: reshapeModulation(
          partAt(parts, 2, "FluxModulation.modulate"),
          batch,
          this.#hiddenSize,
        ),
      };
      if (!this.#double) {
        return { kind: "single", attention };
      }
      return {
        kind: "double",
        attention,
        feedForward: {
          shift: reshapeModulation(
            partAt(parts, 3, "FluxModulation.modulate"),
            batch,
            this.#hiddenSize,
          ),
          scale: reshapeModulation(
            partAt(parts, 4, "FluxModulation.modulate"),
            batch,
            this.#hiddenSize,
          ),
          gate: reshapeModulation(
            partAt(parts, 5, "FluxModulation.modulate"),
            batch,
            this.#hiddenSize,
          ),
        },
      };
    } finally {
      freeArrays(parts);
    }
  }
}

/** FLUX MLP block used by the double-stream transformer layers. */
export class FluxFeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(hiddenSize: number, mlpHiddenSize: number) {
    super();
    this.linear1 = new Linear(hiddenSize, mlpHiddenSize);
    this.linear2 = new Linear(mlpHiddenSize, hiddenSize);
  }

  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}

/** Joint image/text FLUX block with separate stream projections and residuals. */
export class FluxDoubleStreamBlock extends Module {
  imageModulation: FluxModulation;
  imageAttention: FluxSelfAttention;
  imageFeedForward: FluxFeedForward;
  textModulation: FluxModulation;
  textAttention: FluxSelfAttention;
  textFeedForward: FluxFeedForward;
  #hiddenSize: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    mlpHiddenSize: number;
    qkvBias: boolean;
  }) {
    super();
    this.imageModulation = new FluxModulation(options.hiddenSize, true);
    this.imageAttention = new FluxSelfAttention(
      options.hiddenSize,
      options.numHeads,
      options.headDim,
      options.qkvBias,
    );
    this.imageFeedForward = new FluxFeedForward(options.hiddenSize, options.mlpHiddenSize);
    this.textModulation = new FluxModulation(options.hiddenSize, true);
    this.textAttention = new FluxSelfAttention(
      options.hiddenSize,
      options.numHeads,
      options.headDim,
      options.qkvBias,
    );
    this.textFeedForward = new FluxFeedForward(options.hiddenSize, options.mlpHiddenSize);
    this.#hiddenSize = options.hiddenSize;
  }

  forward(image: MxArray, text: MxArray, vec: MxArray, rope: MxArray): MxArray {
    const next = this.run(image, text, vec, rope);
    next.text.free();
    return next.image;
  }

  run(
    image: MxArray,
    text: MxArray,
    vec: MxArray,
    rope: MxArray,
  ): { image: MxArray; text: MxArray } {
    const imageModulation = this.imageModulation.modulate(vec);
    const textModulation = this.textModulation.modulate(vec);
    if (imageModulation.kind !== "double" || textModulation.kind !== "double") {
      disposeModulation(imageModulation);
      disposeModulation(textModulation);
      throw new Error("FluxDoubleStreamBlock.forward: expected double modulation.");
    }
    const imageProjection = this.#projectStream(
      image,
      this.imageAttention,
      imageModulation.attention,
      "FluxDoubleStreamBlock.forward image",
    );
    const textProjection = this.#projectStream(
      text,
      this.textAttention,
      textModulation.attention,
      "FluxDoubleStreamBlock.forward text",
    );
    try {
      using queries = concatenate([textProjection.queries, imageProjection.queries], 2);
      using keys = concatenate([textProjection.keys, imageProjection.keys], 2);
      using values = concatenate([textProjection.values, imageProjection.values], 2);
      using attended = fluxAttention(queries, keys, values, rope);
      using textAttention = sliceAxis(attended, 1, 0, text.shape[1] ?? 0);
      using imageAttention = sliceAxis(attended, 1, text.shape[1] ?? 0, attended.shape[1] ?? 0);
      using projectedText = this.textAttention.projectOutput(textAttention);
      using projectedImage = this.imageAttention.projectOutput(imageAttention);
      using gatedTextAttention = multiply(textModulation.attention.gate, projectedText);
      using gatedImageAttention = multiply(imageModulation.attention.gate, projectedImage);
      using textAttentionResidual = add(text, gatedTextAttention);
      using imageAttentionResidual = add(image, gatedImageAttention);
      using nextText = this.#feedForwardResidual(
        textAttentionResidual,
        textModulation.feedForward,
        this.textFeedForward,
        "FluxDoubleStreamBlock.forward text mlp",
      );
      using nextImage = this.#feedForwardResidual(
        imageAttentionResidual,
        imageModulation.feedForward,
        this.imageFeedForward,
        "FluxDoubleStreamBlock.forward image mlp",
      );
      return { text: retainArray(nextText), image: retainArray(nextImage) };
    } finally {
      disposeProjection(imageProjection);
      disposeProjection(textProjection);
      disposeModulation(imageModulation);
      disposeModulation(textModulation);
    }
  }

  #projectStream(
    hiddenStates: MxArray,
    attention: FluxSelfAttention,
    modulation: FluxModulationTriplet,
    owner: string,
  ): FluxAttentionProjection {
    using normalized = affineFreeLayerNorm(hiddenStates, this.#hiddenSize, owner);
    using modulated = applyModulation(normalized, modulation.shift, modulation.scale);
    return attention.project(modulated);
  }

  #feedForwardResidual(
    hiddenStates: MxArray,
    modulation: FluxModulationTriplet,
    feedForward: FluxFeedForward,
    owner: string,
  ): MxArray {
    using normalized = affineFreeLayerNorm(hiddenStates, this.#hiddenSize, owner);
    using modulated = applyModulation(normalized, modulation.shift, modulation.scale);
    using mlp = feedForward.forward(modulated);
    using gated = multiply(modulation.gate, mlp);
    return add(hiddenStates, gated);
  }
}

/** Single-stream FLUX block after text and image sequences are concatenated. */
export class FluxSingleStreamBlock extends Module {
  modulation: FluxModulation;
  linear1: Linear;
  linear2: Linear;
  #hiddenSize: number;
  #numHeads: number;
  #headDim: number;
  #mlpHiddenSize: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    mlpHiddenSize: number;
    qkvBias: boolean;
  }) {
    super();
    this.modulation = new FluxModulation(options.hiddenSize, false);
    this.linear1 = new Linear(
      options.hiddenSize,
      options.hiddenSize * 3 + options.mlpHiddenSize,
      options.qkvBias,
    );
    this.linear2 = new Linear(options.hiddenSize + options.mlpHiddenSize, options.hiddenSize);
    this.#hiddenSize = options.hiddenSize;
    this.#numHeads = options.numHeads;
    this.#headDim = options.headDim;
    this.#mlpHiddenSize = options.mlpHiddenSize;
  }

  forward(hiddenStates: MxArray, vec: MxArray, rope: MxArray): MxArray {
    const modulation = this.modulation.modulate(vec);
    if (modulation.kind !== "single") {
      disposeModulation(modulation);
      throw new Error("FluxSingleStreamBlock.forward: expected single modulation.");
    }
    try {
      const { batch, length } = assertSequence3d(hiddenStates, "FluxSingleStreamBlock.forward");
      using normalized = affineFreeLayerNorm(
        hiddenStates,
        this.#hiddenSize,
        "FluxSingleStreamBlock.forward",
      );
      using modulated = applyModulation(
        normalized,
        modulation.attention.shift,
        modulation.attention.scale,
      );
      using projected = this.linear1.forward(modulated);
      using rawQueries = sliceLastAxis(projected, 0, this.#hiddenSize);
      using rawKeys = sliceLastAxis(projected, this.#hiddenSize, this.#hiddenSize * 2);
      using rawValues = sliceLastAxis(projected, this.#hiddenSize * 2, this.#hiddenSize * 3);
      using mlpHidden = sliceLastAxis(
        projected,
        this.#hiddenSize * 3,
        this.#hiddenSize * 3 + this.#mlpHiddenSize,
      );
      using queries = this.#reshapeAttention(rawQueries, batch, length);
      using keys = this.#reshapeAttention(rawKeys, batch, length);
      using values = this.#reshapeAttention(rawValues, batch, length);
      using attention = fluxAttention(queries, keys, values, rope);
      using activatedMlp = geluApprox(mlpHidden);
      using combined = concatenate([attention, activatedMlp], -1);
      using projectedOutput = this.linear2.forward(combined);
      using gated = multiply(modulation.attention.gate, projectedOutput);
      return add(hiddenStates, gated);
    } finally {
      disposeModulation(modulation);
    }
  }

  #reshapeAttention(x: MxArray, batch: number, length: number): MxArray {
    using reshaped = reshape(x, [batch, length, this.#numHeads, this.#headDim]);
    return transpose(reshaped, [0, 2, 1, 3]);
  }
}

/** Final FLUX AdaLN projection back to packed latent channels. */
export class FluxLastLayer extends Module {
  modulation: Linear;
  projection: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number, outChannels: number) {
    super();
    this.modulation = new Linear(hiddenSize, hiddenSize * 2);
    this.projection = new Linear(hiddenSize, outChannels);
    this.#hiddenSize = hiddenSize;
  }

  forward(hiddenStates: MxArray, vec: MxArray): MxArray {
    const [batch, channels] = vec.shape;
    if (vec.shape.length !== 2 || batch === undefined || channels !== this.#hiddenSize) {
      throw new Error("FluxLastLayer.forward: vec must have shape [batch, hiddenSize].");
    }
    using activated = silu(vec);
    using projected = this.modulation.forward(activated);
    const parts = split(projected, 2, -1);
    try {
      const shift = partAt(parts, 0, "FluxLastLayer.forward");
      const scale = partAt(parts, 1, "FluxLastLayer.forward");
      using reshapedShift = reshapeModulation(shift, batch, this.#hiddenSize);
      using reshapedScale = reshapeModulation(scale, batch, this.#hiddenSize);
      using normalized = affineFreeLayerNorm(
        hiddenStates,
        this.#hiddenSize,
        "FluxLastLayer.forward",
      );
      using modulated = applyModulation(normalized, reshapedShift, reshapedScale);
      return this.projection.forward(modulated);
    } finally {
      freeArrays(parts);
    }
  }
}
