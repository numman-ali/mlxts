import type { MxArray } from "@mlxts/core";
import {
  add,
  fastRmsNorm,
  geluApprox,
  multiply,
  random,
  reshape,
  retainArray,
  split,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { LtxVideoAttention, type LtxVideoAttentionOptions } from "./attention";
import type { LtxVideoTransformerConfig } from "./config";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { applyScaleShift, assertSequence3d, freeArrays } from "./tensor-utils";

type LtxVideoBlockModulation = {
  shiftMsa: MxArray;
  scaleMsa: MxArray;
  gateMsa: MxArray;
  shiftMlp: MxArray;
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

function disposeBlockModulation(modulation: LtxVideoBlockModulation): void {
  modulation.shiftMsa.free();
  modulation.scaleMsa.free();
  modulation.gateMsa.free();
  modulation.shiftMlp.free();
  modulation.scaleMlp.free();
  modulation.gateMlp.free();
}

function scaledNormal(shape: number[], scale: number): MxArray {
  using values = random.normal(shape);
  return multiply(values, scale);
}

/** Diffusers GELU-approximate feed-forward layer used by LTX blocks. */
export class LtxVideoFeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(hiddenSize: number) {
    super();
    this.linear1 = new Linear(hiddenSize, hiddenSize * 4);
    this.linear2 = new Linear(hiddenSize * 4, hiddenSize);
  }

  /** Run the LTX feed-forward projection. */
  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}

/** One classic LTX-Video transformer block. */
export class LtxVideoTransformerBlock extends Module {
  norm1: null;
  attn1: LtxVideoAttention;
  norm2: null;
  attn2: LtxVideoAttention;
  ff: LtxVideoFeedForward;
  scaleShiftTable: MxArray;
  #hiddenSize: number;
  #normEps: number;

  constructor(config: LtxVideoTransformerConfig) {
    super();
    this.norm1 = null;
    this.attn1 = new LtxVideoAttention({
      hiddenSize: config.hiddenSize,
      numHeads: config.numAttentionHeads,
      headDim: config.attentionHeadDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
    });
    this.norm2 = null;
    this.attn2 = new LtxVideoAttention({
      hiddenSize: config.hiddenSize,
      numHeads: config.numAttentionHeads,
      headDim: config.attentionHeadDim,
      crossAttentionDim: config.crossAttentionDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
    });
    this.ff = new LtxVideoFeedForward(config.hiddenSize);
    this.scaleShiftTable = scaledNormal([6, config.hiddenSize], config.hiddenSize ** -0.5);
    this.#hiddenSize = config.hiddenSize;
    this.#normEps = config.normEps;
  }

  forward(_hiddenStates: MxArray): MxArray {
    throw new Error("LtxVideoTransformerBlock.forward: use run() inside the LTX transformer.");
  }

  /** Run one self-attention, cross-attention, and feed-forward block. */
  run(
    hiddenStates: MxArray,
    encoderHiddenStates: MxArray,
    timestepModulation: MxArray,
    imageRotaryEmbeddings: LtxRotaryEmbeddings,
    encoderAttentionMask?: MxArray,
  ): MxArray {
    const hiddenShape = assertSequence3d(hiddenStates, "LtxVideoTransformerBlock.run hiddenStates");
    if (hiddenShape.channels !== this.#hiddenSize) {
      throw new Error("LtxVideoTransformerBlock.run: hidden size mismatch.");
    }
    const modulation = this.#modulation(timestepModulation, hiddenShape.batch);
    try {
      using normHidden = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
      using selfAttentionInput = applyScaleShift(
        normHidden,
        modulation.shiftMsa,
        modulation.scaleMsa,
      );
      using selfAttention = this.attn1.run(selfAttentionInput, { imageRotaryEmbeddings });
      using gatedSelfAttention = multiply(selfAttention, modulation.gateMsa);
      using selfResidual = add(hiddenStates, gatedSelfAttention);
      const crossAttentionOptions: LtxVideoAttentionOptions = { encoderHiddenStates };
      if (encoderAttentionMask !== undefined) {
        crossAttentionOptions.encoderAttentionMask = encoderAttentionMask;
      }
      using crossAttention = this.attn2.run(selfResidual, crossAttentionOptions);
      using crossResidual = add(selfResidual, crossAttention);
      using normFeedForward = fastRmsNorm(crossResidual, undefined, { eps: this.#normEps });
      using feedForwardInput = applyScaleShift(
        normFeedForward,
        modulation.shiftMlp,
        modulation.scaleMlp,
      );
      using feedForward = this.ff.forward(feedForwardInput);
      using gatedFeedForward = multiply(feedForward, modulation.gateMlp);
      using output = add(crossResidual, gatedFeedForward);
      return retainArray(output);
    } finally {
      disposeBlockModulation(modulation);
    }
  }

  #modulation(timestepModulation: MxArray, batch: number): LtxVideoBlockModulation {
    if (
      timestepModulation.shape.length !== 3 ||
      timestepModulation.shape[0] !== batch ||
      timestepModulation.shape[1] !== 6 ||
      timestepModulation.shape[2] !== this.#hiddenSize
    ) {
      throw new Error("LtxVideoTransformerBlock.modulation: timestep modulation shape mismatch.");
    }
    using table = reshape(this.scaleShiftTable, [1, 6, this.#hiddenSize]);
    using values = add(table, timestepModulation);
    const parts = split(values, 6, 1);
    try {
      return {
        shiftMsa: retainArray(partAt(parts, 0, "LtxVideoTransformerBlock.modulation")),
        scaleMsa: retainArray(partAt(parts, 1, "LtxVideoTransformerBlock.modulation")),
        gateMsa: retainArray(partAt(parts, 2, "LtxVideoTransformerBlock.modulation")),
        shiftMlp: retainArray(partAt(parts, 3, "LtxVideoTransformerBlock.modulation")),
        scaleMlp: retainArray(partAt(parts, 4, "LtxVideoTransformerBlock.modulation")),
        gateMlp: retainArray(partAt(parts, 5, "LtxVideoTransformerBlock.modulation")),
      };
    } finally {
      freeArrays(parts);
    }
  }
}
