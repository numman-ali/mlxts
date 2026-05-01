import type { MxArray } from "@mlxts/core";
import { add, geluApprox, multiply, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import {
  StableDiffusion3JointAttention,
  type StableDiffusion3JointAttentionOutput,
  StableDiffusion3SelfAttention,
} from "./attention";
import type { StableDiffusion3QkNorm } from "./config";
import {
  disposeStableDiffusion3AdaLayerNormZero,
  disposeStableDiffusion35AdaLayerNormZeroX,
  StableDiffusion3AdaLayerNormContinuous,
  StableDiffusion3AdaLayerNormZero,
  type StableDiffusion3AdaLayerNormZeroOutput,
  StableDiffusion35AdaLayerNormZeroX,
  type StableDiffusion35AdaLayerNormZeroXOutput,
} from "./normalization";
import { affineFreeLayerNorm, applyScaleShift } from "./tensor-utils";

export type StableDiffusion3BlockOutput = {
  hiddenStates: MxArray;
  encoderHiddenStates: MxArray | null;
};

function isAdaLayerNormZeroXOutput(
  output: StableDiffusion3AdaLayerNormZeroOutput | StableDiffusion35AdaLayerNormZeroXOutput,
): output is StableDiffusion35AdaLayerNormZeroXOutput {
  return "hiddenStates2" in output && "gateMsa2" in output;
}

/** SD3 GELU feed-forward block. */
export class StableDiffusion3FeedForward extends Module {
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

/** MMDiT block used by Stable Diffusion 3 and 3.5 transformers. */
export class StableDiffusion3JointTransformerBlock extends Module {
  norm1: StableDiffusion3AdaLayerNormZero | StableDiffusion35AdaLayerNormZeroX;
  norm1Context: StableDiffusion3AdaLayerNormZero | StableDiffusion3AdaLayerNormContinuous;
  attention: StableDiffusion3JointAttention;
  attention2: StableDiffusion3SelfAttention | null;
  ff: StableDiffusion3FeedForward;
  ffContext: StableDiffusion3FeedForward | null;
  #hiddenSize: number;
  #contextPreOnly: boolean;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    qkNorm: StableDiffusion3QkNorm;
    contextPreOnly: boolean;
    useDualAttention: boolean;
  }) {
    super();
    this.norm1 = options.useDualAttention
      ? new StableDiffusion35AdaLayerNormZeroX(options.hiddenSize)
      : new StableDiffusion3AdaLayerNormZero(options.hiddenSize);
    this.norm1Context = options.contextPreOnly
      ? new StableDiffusion3AdaLayerNormContinuous(options.hiddenSize, options.hiddenSize)
      : new StableDiffusion3AdaLayerNormZero(options.hiddenSize);
    this.attention = new StableDiffusion3JointAttention({
      hiddenSize: options.hiddenSize,
      numHeads: options.numHeads,
      headDim: options.headDim,
      qkNorm: options.qkNorm,
      contextPreOnly: options.contextPreOnly,
    });
    this.attention2 = options.useDualAttention
      ? new StableDiffusion3SelfAttention(
          options.hiddenSize,
          options.numHeads,
          options.headDim,
          options.qkNorm,
        )
      : null;
    this.ff = new StableDiffusion3FeedForward(options.hiddenSize, options.hiddenSize * 4);
    this.ffContext = options.contextPreOnly
      ? null
      : new StableDiffusion3FeedForward(options.hiddenSize, options.hiddenSize * 4);
    this.#hiddenSize = options.hiddenSize;
    this.#contextPreOnly = options.contextPreOnly;
  }

  forward(_hiddenStates: MxArray): MxArray {
    throw new Error(
      "StableDiffusion3JointTransformerBlock.forward: use run() inside the transformer.",
    );
  }

  /** Run one SD3 MMDiT block over image and context streams. */
  run(
    hiddenStates: MxArray,
    encoderHiddenStates: MxArray,
    temb: MxArray,
  ): StableDiffusion3BlockOutput {
    const imageModulation =
      this.norm1 instanceof StableDiffusion35AdaLayerNormZeroX
        ? this.norm1.modulate(hiddenStates, temb)
        : this.norm1.modulate(hiddenStates, temb);
    let contextModulation: StableDiffusion3AdaLayerNormZeroOutput | null = null;
    let contextAttentionInput: MxArray | null = null;
    try {
      if (this.#contextPreOnly) {
        if (!(this.norm1Context instanceof StableDiffusion3AdaLayerNormContinuous)) {
          throw new Error(
            "StableDiffusion3JointTransformerBlock.run: expected continuous context norm.",
          );
        }
        contextAttentionInput = this.norm1Context.forward(encoderHiddenStates, temb);
      } else {
        if (!(this.norm1Context instanceof StableDiffusion3AdaLayerNormZero)) {
          throw new Error(
            "StableDiffusion3JointTransformerBlock.run: expected context AdaLN-Zero.",
          );
        }
        contextModulation = this.norm1Context.modulate(encoderHiddenStates, temb);
        contextAttentionInput = contextModulation.hiddenStates;
      }

      const attentionOutput = this.attention.run(
        imageModulation.hiddenStates,
        contextAttentionInput,
      );
      try {
        const hiddenAfterAttention = this.#imageAttentionResidual(
          hiddenStates,
          imageModulation,
          attentionOutput.hidden,
        );
        try {
          const nextHidden = this.#feedForwardResidual(
            hiddenAfterAttention,
            imageModulation,
            this.ff,
            "StableDiffusion3JointTransformerBlock.run image ff",
          );
          try {
            return {
              hiddenStates: nextHidden,
              encoderHiddenStates: this.#contextResidual(
                encoderHiddenStates,
                contextModulation,
                attentionOutput,
              ),
            };
          } catch (error) {
            nextHidden.free();
            throw error;
          }
        } finally {
          hiddenAfterAttention.free();
        }
      } finally {
        attentionOutput.hidden.free();
        attentionOutput.context?.free();
      }
    } finally {
      if (this.#contextPreOnly) {
        contextAttentionInput?.free();
      }
      if (isAdaLayerNormZeroXOutput(imageModulation)) {
        disposeStableDiffusion35AdaLayerNormZeroX(imageModulation);
      } else {
        disposeStableDiffusion3AdaLayerNormZero(imageModulation);
      }
      if (contextModulation !== null) {
        disposeStableDiffusion3AdaLayerNormZero(contextModulation);
      }
    }
  }

  #imageAttentionResidual(
    hiddenStates: MxArray,
    modulation: StableDiffusion3AdaLayerNormZeroOutput | StableDiffusion35AdaLayerNormZeroXOutput,
    attentionHidden: MxArray,
  ): MxArray {
    using gated = multiply(modulation.gateMsa, attentionHidden);
    using firstResidual = add(hiddenStates, gated);
    if (!("hiddenStates2" in modulation)) {
      return retainArray(firstResidual);
    }
    if (this.attention2 === null) {
      throw new Error("StableDiffusion3JointTransformerBlock.run: missing dual attention.");
    }
    using dualAttention = this.attention2.forward(modulation.hiddenStates2);
    using gatedDual = multiply(modulation.gateMsa2, dualAttention);
    return add(firstResidual, gatedDual);
  }

  #feedForwardResidual(
    hiddenStates: MxArray,
    modulation: Pick<StableDiffusion3AdaLayerNormZeroOutput, "shiftMlp" | "scaleMlp" | "gateMlp">,
    feedForward: StableDiffusion3FeedForward,
    owner: string,
  ): MxArray {
    using normalized = affineFreeLayerNorm(hiddenStates, this.#hiddenSize, owner);
    using modulated = applyScaleShift(normalized, modulation.shiftMlp, modulation.scaleMlp);
    using output = feedForward.forward(modulated);
    using gated = multiply(modulation.gateMlp, output);
    return add(hiddenStates, gated);
  }

  #contextResidual(
    encoderHiddenStates: MxArray,
    contextModulation: StableDiffusion3AdaLayerNormZeroOutput | null,
    attentionOutput: StableDiffusion3JointAttentionOutput,
  ): MxArray | null {
    if (this.#contextPreOnly) {
      return null;
    }
    if (contextModulation === null || attentionOutput.context === null || this.ffContext === null) {
      throw new Error("StableDiffusion3JointTransformerBlock.run: missing context stream.");
    }
    using gated = multiply(contextModulation.gateMsa, attentionOutput.context);
    using attentionResidual = add(encoderHiddenStates, gated);
    return this.#feedForwardResidual(
      attentionResidual,
      contextModulation,
      this.ffContext,
      "StableDiffusion3JointTransformerBlock.run context ff",
    );
  }
}
