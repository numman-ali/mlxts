/**
 * Gemma 3 text decoder block.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import type { AttentionMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import { Gemma3Attention } from "./attention";
import { Gemma3MLP } from "./mlp";
import { Gemma3RMSNorm } from "./norm";
import type { Gemma3TextConfig } from "./types";

/** Decoder block with Gemma 3's q/k-norm attention and feedforward norm pattern. */
export class Gemma3DecoderBlock extends Module {
  inputLayerNorm: Gemma3RMSNorm;
  selfAttention: Gemma3Attention;
  postAttentionLayerNorm: Gemma3RMSNorm;
  preFeedforwardLayerNorm: Gemma3RMSNorm;
  mlp: Gemma3MLP;
  postFeedforwardLayerNorm: Gemma3RMSNorm;
  #isSliding: boolean;

  constructor(config: Gemma3TextConfig, layerIndex: number) {
    super();
    this.inputLayerNorm = new Gemma3RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.selfAttention = new Gemma3Attention(config, layerIndex);
    this.postAttentionLayerNorm = new Gemma3RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.preFeedforwardLayerNorm = new Gemma3RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.mlp = new Gemma3MLP(config);
    this.postFeedforwardLayerNorm = new Gemma3RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.#isSliding = this.selfAttention.layerType === "sliding_attention";
  }

  get isSliding(): boolean {
    return this.#isSliding;
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, cache?: TransformerCache, attentionMask?: AttentionMask): MxArray {
    using normalizedForAttention = this.inputLayerNorm.forward(x);
    using attentionOutput = this.selfAttention.run(normalizedForAttention, cache, attentionMask);
    using normalizedAttentionOutput = this.postAttentionLayerNorm.forward(attentionOutput);
    using residualAfterAttention = add(x, normalizedAttentionOutput);
    using normalizedForMlp = this.preFeedforwardLayerNorm.forward(residualAfterAttention);
    using mlpOutput = this.mlp.forward(normalizedForMlp);
    using normalizedMlpOutput = this.postFeedforwardLayerNorm.forward(mlpOutput);
    return add(residualAfterAttention, normalizedMlpOutput);
  }
}
