/**
 * Gemma 4 dense text decoder block.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, multiply, ones } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import { gegluApprox } from "../../infrastructure/gated-activations";
import type { AttentionMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
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
  mlp: Gemma4TextMLP;
  postFeedforwardLayerNorm: Gemma4RMSNorm;
  layerScalar: MxArray;
  perLayerInputGate: Linear | null;
  perLayerProjection: Linear | null;
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
      using hiddenAfterMlp = this.runFeedforwardTail(x, attentionOutput);
      if (perLayerInput !== undefined) {
        using hiddenWithPerLayerInput = this.applyPerLayerInput(hiddenAfterMlp, perLayerInput);
        return { hidden: multiply(hiddenWithPerLayerInput, this.layerScalar), keyValues };
      }
      return { hidden: multiply(hiddenAfterMlp, this.layerScalar), keyValues };
    } catch (error) {
      keyValues?.keys.free();
      keyValues?.values.free();
      throw error;
    } finally {
      attentionOutput.free();
    }
  }

  private runFeedforwardTail(residualInput: MxArray, attentionOutput: MxArray): MxArray {
    using normalizedAttentionOutput = this.postAttentionLayerNorm.forward(attentionOutput);
    using residualAfterAttention = add(residualInput, normalizedAttentionOutput);
    using normalizedForMlp = this.preFeedforwardLayerNorm.forward(residualAfterAttention);
    using mlpOutput = this.mlp.forward(normalizedForMlp);
    using normalizedMlpOutput = this.postFeedforwardLayerNorm.forward(mlpOutput);
    return add(residualAfterAttention, normalizedMlpOutput);
  }

  private applyPerLayerInput(hidden: MxArray, perLayerInput: MxArray): MxArray {
    if (
      this.perLayerInputGate === null ||
      this.perLayerProjection === null ||
      this.postPerLayerInputNorm === null
    ) {
      throw new Error("Gemma4TextDecoderBlock: expected per-layer input modules to be enabled.");
    }

    using gatedInput = this.perLayerInputGate.forward(hidden);
    using gatedPerLayerInput = gegluApprox(gatedInput, perLayerInput);
    using projectedPerLayerInput = this.perLayerProjection.forward(gatedPerLayerInput);
    using normalizedPerLayerInput = this.postPerLayerInputNorm.forward(projectedPerLayerInput);
    return add(hidden, normalizedPerLayerInput);
  }
}
