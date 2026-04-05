/**
 * Gemma 4 dense text decoder block.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, geluApprox, multiply, ones } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { AttentionMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import type { ForwardModule } from "../llama-like/types";
import { Gemma4TextAttention } from "./attention";
import { Gemma4TextMLP } from "./mlp";
import { Gemma4RMSNorm } from "./norm";
import type { Gemma4SharedKeyValues, Gemma4TextConfig } from "./types";

/** Dense Gemma 4 decoder block with mixed local/global attention. */
export class Gemma4TextDecoderBlock extends Module {
  inputLayerNorm: Gemma4RMSNorm;
  selfAttention: Gemma4TextAttention;
  postAttentionLayerNorm: Gemma4RMSNorm;
  preFeedforwardLayerNorm: Gemma4RMSNorm;
  mlp: ForwardModule;
  postFeedforwardLayerNorm: Gemma4RMSNorm;
  layerScalar: MxArray;
  perLayerInputGate: ForwardModule | null;
  perLayerProjection: ForwardModule | null;
  postPerLayerInputNorm: Gemma4RMSNorm | null;

  constructor(config: Gemma4TextConfig, layerIndex: number) {
    super();
    this.inputLayerNorm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.selfAttention = new Gemma4TextAttention(config, layerIndex);
    this.postAttentionLayerNorm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.preFeedforwardLayerNorm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.mlp = new Gemma4TextMLP(config, layerIndex);
    this.postFeedforwardLayerNorm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.layerScalar = ones([1]);
    if (config.hiddenSizePerLayerInput > 0) {
      this.perLayerInputGate = new Linear(config.hiddenSize, config.hiddenSizePerLayerInput, false);
      this.perLayerProjection = new Linear(
        config.hiddenSizePerLayerInput,
        config.hiddenSize,
        false,
      );
      this.postPerLayerInputNorm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    } else {
      this.perLayerInputGate = null;
      this.perLayerProjection = null;
      this.postPerLayerInputNorm = null;
    }
  }

  forward(x: MxArray): MxArray {
    const result = this.run(x);
    result.keyValues?.keys.free();
    result.keyValues?.values.free();
    return result.hidden;
  }

  run(
    x: MxArray,
    cache?: TransformerCache,
    sharedKeyValues?: Gemma4SharedKeyValues,
    perLayerInput?: MxArray,
    attentionMask?: AttentionMask,
  ): { hidden: MxArray; keyValues: Gemma4SharedKeyValues | null } {
    using normalizedForAttention = this.inputLayerNorm.forward(x);
    const { output: attentionOutput, keyValues } = this.selfAttention.run(
      normalizedForAttention,
      cache,
      sharedKeyValues,
      attentionMask,
    );

    try {
      using normalizedAttentionOutput = this.postAttentionLayerNorm.forward(attentionOutput);
      using residualAfterAttention = add(x, normalizedAttentionOutput);
      using normalizedForMlp = this.preFeedforwardLayerNorm.forward(residualAfterAttention);
      using mlpOutput = this.mlp.forward(normalizedForMlp);
      using normalizedMlpOutput = this.postFeedforwardLayerNorm.forward(mlpOutput);

      let hidden = add(residualAfterAttention, normalizedMlpOutput);

      try {
        if (
          perLayerInput !== undefined &&
          this.perLayerInputGate !== null &&
          this.perLayerProjection !== null &&
          this.postPerLayerInputNorm !== null
        ) {
          using gatedInput = this.perLayerInputGate.forward(hidden);
          using activatedGate = geluApprox(gatedInput);
          using gatedPerLayerInput = multiply(activatedGate, perLayerInput);
          using projectedPerLayerInput = this.perLayerProjection.forward(gatedPerLayerInput);
          using normalizedPerLayerInput =
            this.postPerLayerInputNorm.forward(projectedPerLayerInput);
          const nextHidden = add(hidden, normalizedPerLayerInput);
          hidden.free();
          hidden = nextHidden;
        }

        const scaledHidden = multiply(hidden, this.layerScalar);
        hidden.free();
        hidden = scaledHidden;
        return { hidden, keyValues };
      } catch (error) {
        hidden.free();
        throw error;
      }
    } catch (error) {
      keyValues?.keys.free();
      keyValues?.values.free();
      throw error;
    } finally {
      attentionOutput.free();
    }
  }
}
