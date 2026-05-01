/**
 * FLUX.2 Klein transformer blocks.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  divide,
  maximum,
  mean,
  minimum,
  multiply,
  reshape,
  retainArray,
  split,
  sqrt,
  square,
  subtract,
} from "@mlxts/core";
import { Linear, Module, silu, swiglu } from "@mlxts/nn";

import { Flux2JointAttention, Flux2ParallelSelfAttention } from "./attention";
import { assertFlux2Sequence3d, freeFlux2Arrays } from "./tensor-utils";

export type Flux2ModulationTriplet = {
  shift: MxArray;
  scale: MxArray;
  gate: MxArray;
};

export type Flux2BlockModulation = {
  attention: Flux2ModulationTriplet;
};

export type Flux2BlockOutput = {
  image: MxArray;
  text: MxArray;
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function disposeTriplet(triplet: Flux2ModulationTriplet): void {
  triplet.shift.free();
  triplet.scale.free();
  triplet.gate.free();
}

export function disposeFlux2Modulation(modulation: Flux2BlockModulation): void {
  disposeTriplet(modulation.attention);
}

function affineFreeLayerNorm(x: MxArray, dims: number, eps: number, owner: string): MxArray {
  const shape = assertFlux2Sequence3d(x, owner);
  if (shape.channels !== dims) {
    throw new Error(`${owner}: expected last dimension ${dims}, got ${shape.channels}.`);
  }
  using center = mean(x, -1, true);
  using centered = subtract(x, center);
  using squared = square(centered);
  using variance = mean(squared, -1, true);
  using stabilized = add(variance, eps);
  using denominator = sqrt(stabilized);
  return divide(centered, denominator);
}

function applyModulation(x: MxArray, modulation: Flux2ModulationTriplet): MxArray {
  using scaled = add(modulation.scale, 1);
  using multiplied = multiply(x, scaled);
  return add(multiplied, modulation.shift);
}

function reshapeModulation(part: MxArray, batch: number, hiddenSize: number): MxArray {
  return reshape(part, [batch, 1, hiddenSize]);
}

function clipFloat16(x: MxArray): MxArray {
  if (x.dtype !== "float16") {
    return retainArray(x);
  }
  using upper = minimum(x, 65504);
  return maximum(upper, -65504);
}

/** Shared FLUX.2 modulation projection used outside the repeated blocks. */
export class Flux2Modulation extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number, sets: number) {
    super();
    if (!Number.isInteger(sets) || sets <= 0) {
      throw new Error("Flux2Modulation: sets must be a positive integer.");
    }
    this.linear = new Linear(hiddenSize, hiddenSize * 3 * sets, false);
    this.#hiddenSize = hiddenSize;
  }

  forward(vec: MxArray): MxArray {
    const [batch, channels] = vec.shape;
    if (vec.shape.length !== 2 || batch === undefined || channels !== this.#hiddenSize) {
      throw new Error("Flux2Modulation.forward: vec must have shape [batch, hiddenSize].");
    }
    using activated = silu(vec);
    return this.linear.forward(activated);
  }
}

export function splitFlux2Modulation(
  projected: MxArray,
  sets: number,
  hiddenSize: number,
  batch: number,
): Flux2BlockModulation[] {
  const [projectedBatch, projectedChannels] = projected.shape;
  if (
    projected.shape.length !== 2 ||
    projectedBatch !== batch ||
    projectedChannels !== hiddenSize * 3 * sets
  ) {
    throw new Error("splitFlux2Modulation: projected modulation has an unexpected shape.");
  }
  const parts = split(projected, sets * 3, -1);
  const modulations: Flux2BlockModulation[] = [];
  try {
    for (let index = 0; index < sets; index += 1) {
      modulations.push({
        attention: {
          shift: reshapeModulation(
            partAt(parts, index * 3, "splitFlux2Modulation"),
            batch,
            hiddenSize,
          ),
          scale: reshapeModulation(
            partAt(parts, index * 3 + 1, "splitFlux2Modulation"),
            batch,
            hiddenSize,
          ),
          gate: reshapeModulation(
            partAt(parts, index * 3 + 2, "splitFlux2Modulation"),
            batch,
            hiddenSize,
          ),
        },
      });
    }
    return modulations;
  } finally {
    freeFlux2Arrays(parts);
  }
}

/** Bias-free SwiGLU feed-forward layer used by FLUX.2 double-stream blocks. */
export class Flux2FeedForward extends Module {
  linearIn: Linear;
  linearOut: Linear;

  constructor(hiddenSize: number, mlpHiddenSize: number) {
    super();
    this.linearIn = new Linear(hiddenSize, mlpHiddenSize * 2, false);
    this.linearOut = new Linear(mlpHiddenSize, hiddenSize, false);
  }

  forward(x: MxArray): MxArray {
    using projected = this.linearIn.forward(x);
    const parts = split(projected, 2, -1);
    try {
      using activated = swiglu(
        partAt(parts, 0, "Flux2FeedForward.forward"),
        partAt(parts, 1, "Flux2FeedForward.forward"),
      );
      return this.linearOut.forward(activated);
    } finally {
      freeFlux2Arrays(parts);
    }
  }
}

/** Final FLUX.2 AdaLayerNormContinuous without affine norm parameters. */
export class Flux2AdaptiveLayerNormContinuous extends Module {
  linear: Linear;
  #hiddenSize: number;
  #eps: number;

  constructor(hiddenSize: number, eps: number) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * 2, false);
    this.#hiddenSize = hiddenSize;
    this.#eps = eps;
  }

  forward(hidden: MxArray, conditioning: MxArray): MxArray {
    using activated = silu(conditioning);
    using projected = this.linear.forward(activated);
    const parts = split(projected, 2, -1);
    try {
      const scale = reshapeModulation(
        partAt(parts, 0, "Flux2AdaptiveLayerNormContinuous.forward"),
        conditioning.shape[0] ?? 0,
        this.#hiddenSize,
      );
      const shift = reshapeModulation(
        partAt(parts, 1, "Flux2AdaptiveLayerNormContinuous.forward"),
        conditioning.shape[0] ?? 0,
        this.#hiddenSize,
      );
      try {
        using normalized = affineFreeLayerNorm(
          hidden,
          this.#hiddenSize,
          this.#eps,
          "Flux2AdaptiveLayerNormContinuous.forward",
        );
        using scaled = add(scale, 1);
        using modulated = multiply(normalized, scaled);
        return add(modulated, shift);
      } finally {
        scale.free();
        shift.free();
      }
    } finally {
      freeFlux2Arrays(parts);
    }
  }
}

/** One FLUX.2 double-stream transformer block. */
export class Flux2TransformerBlock extends Module {
  norm1: null;
  norm1Context: null;
  attn: Flux2JointAttention;
  norm2: null;
  ff: Flux2FeedForward;
  norm2Context: null;
  ffContext: Flux2FeedForward;
  #hiddenSize: number;
  #eps: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    mlpHiddenSize: number;
    eps: number;
  }) {
    super();
    this.norm1 = null;
    this.norm1Context = null;
    this.attn = new Flux2JointAttention(
      options.hiddenSize,
      options.numHeads,
      options.headDim,
      options.eps,
    );
    this.norm2 = null;
    this.ff = new Flux2FeedForward(options.hiddenSize, options.mlpHiddenSize);
    this.norm2Context = null;
    this.ffContext = new Flux2FeedForward(options.hiddenSize, options.mlpHiddenSize);
    this.#hiddenSize = options.hiddenSize;
    this.#eps = options.eps;
  }

  forward(
    image: MxArray,
    text: MxArray,
    imageMod: MxArray,
    textMod: MxArray,
    rope: MxArray,
  ): MxArray {
    const next = this.run(image, text, imageMod, textMod, rope);
    next.text.free();
    return next.image;
  }

  /** Run a double-stream FLUX.2 block over image and text streams. */
  run(
    image: MxArray,
    text: MxArray,
    imageMod: MxArray,
    textMod: MxArray,
    rope: MxArray,
  ): Flux2BlockOutput {
    const batch = image.shape[0] ?? 0;
    const imageModulations = splitFlux2Modulation(imageMod, 2, this.#hiddenSize, batch);
    const textModulations = splitFlux2Modulation(textMod, 2, this.#hiddenSize, batch);
    try {
      const imageAttentionModulation = imageModulations[0];
      const imageFeedForwardModulation = imageModulations[1];
      const textAttentionModulation = textModulations[0];
      const textFeedForwardModulation = textModulations[1];
      if (
        imageAttentionModulation === undefined ||
        imageFeedForwardModulation === undefined ||
        textAttentionModulation === undefined ||
        textFeedForwardModulation === undefined
      ) {
        throw new Error("Flux2TransformerBlock.run: expected two modulation sets.");
      }
      using imageNormed = affineFreeLayerNorm(
        image,
        this.#hiddenSize,
        this.#eps,
        "Flux2TransformerBlock.run image",
      );
      using textNormed = affineFreeLayerNorm(
        text,
        this.#hiddenSize,
        this.#eps,
        "Flux2TransformerBlock.run text",
      );
      using imageAttentionInput = applyModulation(imageNormed, imageAttentionModulation.attention);
      using textAttentionInput = applyModulation(textNormed, textAttentionModulation.attention);
      const attention = this.attn.run(imageAttentionInput, textAttentionInput, rope);
      try {
        using gatedImageAttention = multiply(
          imageAttentionModulation.attention.gate,
          attention.image,
        );
        using gatedTextAttention = multiply(textAttentionModulation.attention.gate, attention.text);
        using imageAttentionResidual = add(image, gatedImageAttention);
        using textAttentionResidual = add(text, gatedTextAttention);
        using imageOut = this.#feedForwardResidual(
          imageAttentionResidual,
          imageFeedForwardModulation.attention,
          this.ff,
          "Flux2TransformerBlock.run image mlp",
        );
        using textOut = this.#feedForwardResidual(
          textAttentionResidual,
          textFeedForwardModulation.attention,
          this.ffContext,
          "Flux2TransformerBlock.run text mlp",
        );
        using clippedText = clipFloat16(textOut);
        return { image: retainArray(imageOut), text: retainArray(clippedText) };
      } finally {
        attention.image.free();
        attention.text.free();
      }
    } finally {
      for (const modulation of imageModulations) {
        disposeFlux2Modulation(modulation);
      }
      for (const modulation of textModulations) {
        disposeFlux2Modulation(modulation);
      }
    }
  }

  #feedForwardResidual(
    hidden: MxArray,
    modulation: Flux2ModulationTriplet,
    feedForward: Flux2FeedForward,
    owner: string,
  ): MxArray {
    using normalized = affineFreeLayerNorm(hidden, this.#hiddenSize, this.#eps, owner);
    using modulated = applyModulation(normalized, modulation);
    using output = feedForward.forward(modulated);
    using gated = multiply(modulation.gate, output);
    return add(hidden, gated);
  }
}

/** One FLUX.2 parallel single-stream transformer block. */
export class Flux2SingleTransformerBlock extends Module {
  norm: null;
  attn: Flux2ParallelSelfAttention;
  #hiddenSize: number;
  #eps: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    mlpHiddenSize: number;
    eps: number;
  }) {
    super();
    this.norm = null;
    this.attn = new Flux2ParallelSelfAttention(
      options.hiddenSize,
      options.numHeads,
      options.headDim,
      options.mlpHiddenSize,
      options.eps,
    );
    this.#hiddenSize = options.hiddenSize;
    this.#eps = options.eps;
  }

  /** Run a single-stream FLUX.2 block over the concatenated text/image stream. */
  forward(hiddenStates: MxArray, modulation: MxArray, rope: MxArray): MxArray {
    const batch = hiddenStates.shape[0] ?? 0;
    const modulations = splitFlux2Modulation(modulation, 1, this.#hiddenSize, batch);
    try {
      const attentionModulation = modulations[0];
      if (attentionModulation === undefined) {
        throw new Error("Flux2SingleTransformerBlock.forward: expected one modulation set.");
      }
      using normalized = affineFreeLayerNorm(
        hiddenStates,
        this.#hiddenSize,
        this.#eps,
        "Flux2SingleTransformerBlock.forward",
      );
      using modulated = applyModulation(normalized, attentionModulation.attention);
      using attention = this.attn.forward(modulated, rope);
      using gated = multiply(attentionModulation.attention.gate, attention);
      using hidden = add(hiddenStates, gated);
      return clipFloat16(hidden);
    } finally {
      for (const mod of modulations) {
        disposeFlux2Modulation(mod);
      }
    }
  }
}
