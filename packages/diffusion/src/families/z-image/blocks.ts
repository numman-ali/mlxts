/**
 * Z-Image transformer blocks.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, expandDims, fastLayerNorm, multiply, split, tanh } from "@mlxts/core";
import { Linear, Module, RMSNorm, silu } from "@mlxts/nn";

import { ZImageSelfAttention } from "./attention";
import { freeArrays } from "./tensor-utils";

type ZImageBlockModulation = {
  scaleAttention: MxArray;
  gateAttention: MxArray;
  scaleMlp: MxArray;
  gateMlp: MxArray;
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function disposeBlockModulation(modulation: ZImageBlockModulation): void {
  modulation.scaleAttention.free();
  modulation.gateAttention.free();
  modulation.scaleMlp.free();
  modulation.gateMlp.free();
}

/** SwiGLU feed-forward layer used by Z-Image blocks. */
export class ZImageFeedForward extends Module {
  w1: Linear;
  w2: Linear;
  w3: Linear;

  constructor(hiddenSize: number) {
    super();
    const hiddenDim = Math.floor((hiddenSize / 3) * 8);
    this.w1 = new Linear(hiddenSize, hiddenDim, false);
    this.w2 = new Linear(hiddenDim, hiddenSize, false);
    this.w3 = new Linear(hiddenSize, hiddenDim, false);
  }

  /** Run Z-Image SwiGLU feed-forward projection. */
  forward(x: MxArray): MxArray {
    using gate = this.w1.forward(x);
    using activated = silu(gate);
    using value = this.w3.forward(x);
    using hidden = multiply(activated, value);
    return this.w2.forward(hidden);
  }
}

/** Single-stream Z-Image transformer block. */
export class ZImageTransformerBlock extends Module {
  attention: ZImageSelfAttention;
  feedForward: ZImageFeedForward;
  attentionNorm1: RMSNorm;
  attentionNorm2: RMSNorm;
  ffnNorm1: RMSNorm;
  ffnNorm2: RMSNorm;
  adaLnModulation: Linear | null;
  #hiddenSize: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    normEps: number;
    qkNorm: boolean;
    modulation: boolean;
    adalnDims: number;
  }) {
    super();
    this.attention = new ZImageSelfAttention(
      options.hiddenSize,
      options.numHeads,
      options.qkNorm,
      options.normEps,
    );
    this.feedForward = new ZImageFeedForward(options.hiddenSize);
    this.attentionNorm1 = new RMSNorm(options.hiddenSize, options.normEps);
    this.attentionNorm2 = new RMSNorm(options.hiddenSize, options.normEps);
    this.ffnNorm1 = new RMSNorm(options.hiddenSize, options.normEps);
    this.ffnNorm2 = new RMSNorm(options.hiddenSize, options.normEps);
    this.adaLnModulation = options.modulation
      ? new Linear(options.adalnDims, options.hiddenSize * 4)
      : null;
    this.#hiddenSize = options.hiddenSize;
  }

  /** Run one Z-Image block over a padded hidden sequence. */
  forward(
    x: MxArray,
    rope: MxArray,
    adalnInput?: MxArray,
    attentionMask: MxArray | null = null,
  ): MxArray {
    if (this.adaLnModulation === null) {
      return this.#forwardUnmodulated(x, rope, attentionMask);
    }
    if (adalnInput === undefined) {
      throw new Error("ZImageTransformerBlock.forward: adalnInput is required.");
    }
    const modulation = this.#modulate(adalnInput);
    try {
      using attentionInputNorm = this.attentionNorm1.forward(x);
      using attentionInput = multiply(attentionInputNorm, modulation.scaleAttention);
      using attentionOut = this.attention.forward(attentionInput, rope, attentionMask);
      using attentionOutNorm = this.attentionNorm2.forward(attentionOut);
      using gatedAttention = multiply(modulation.gateAttention, attentionOutNorm);
      using attentionResidual = add(x, gatedAttention);
      using mlpInputNorm = this.ffnNorm1.forward(attentionResidual);
      using mlpInput = multiply(mlpInputNorm, modulation.scaleMlp);
      using mlpOut = this.feedForward.forward(mlpInput);
      using mlpOutNorm = this.ffnNorm2.forward(mlpOut);
      using gatedMlp = multiply(modulation.gateMlp, mlpOutNorm);
      return add(attentionResidual, gatedMlp);
    } finally {
      disposeBlockModulation(modulation);
    }
  }

  #forwardUnmodulated(x: MxArray, rope: MxArray, attentionMask: MxArray | null): MxArray {
    using attentionInput = this.attentionNorm1.forward(x);
    using attentionOut = this.attention.forward(attentionInput, rope, attentionMask);
    using attentionOutNorm = this.attentionNorm2.forward(attentionOut);
    using attentionResidual = add(x, attentionOutNorm);
    using mlpInput = this.ffnNorm1.forward(attentionResidual);
    using mlpOut = this.feedForward.forward(mlpInput);
    using mlpOutNorm = this.ffnNorm2.forward(mlpOut);
    return add(attentionResidual, mlpOutNorm);
  }

  #modulate(adalnInput: MxArray): ZImageBlockModulation {
    if (this.adaLnModulation === null) {
      throw new Error("ZImageTransformerBlock.modulate: block is unmodulated.");
    }
    using projected = this.adaLnModulation.forward(adalnInput);
    using expanded = expandDims(projected, 1);
    const parts = split(expanded, 4, -1);
    try {
      return {
        scaleAttention: add(partAt(parts, 0, "ZImageTransformerBlock.modulate"), 1),
        gateAttention: tanh(partAt(parts, 1, "ZImageTransformerBlock.modulate")),
        scaleMlp: add(partAt(parts, 2, "ZImageTransformerBlock.modulate"), 1),
        gateMlp: tanh(partAt(parts, 3, "ZImageTransformerBlock.modulate")),
      };
    } finally {
      freeArrays(parts);
    }
  }

  get hiddenSize(): number {
    return this.#hiddenSize;
  }
}

/** Final Z-Image AdaLN projection into latent patch channels. */
export class ZImageFinalLayer extends Module {
  modulation: Linear;
  output: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number, adalnDims: number, outChannels: number) {
    super();
    this.modulation = new Linear(adalnDims, hiddenSize);
    this.output = new Linear(hiddenSize, outChannels);
    this.#hiddenSize = hiddenSize;
  }

  /** Project final hidden states into patchified latent velocity channels. */
  forward(x: MxArray, adalnInput: MxArray): MxArray {
    using activated = silu(adalnInput);
    using projected = this.modulation.forward(activated);
    using scale = add(projected, 1);
    using scaleSequence = expandDims(scale, 1);
    using normalized = fastLayerNorm(x, undefined, undefined, { eps: 1e-6 });
    using modulated = multiply(normalized, scaleSequence);
    return this.output.forward(modulated);
  }

  get hiddenSize(): number {
    return this.#hiddenSize;
  }
}
