/**
 * T5 encoder block.
 * @module
 */

import { add, type MxArray } from "@mlxts/core";
import { Module, RMSNorm } from "@mlxts/nn";

import { T5Attention } from "./attention";
import { T5DenseActivationLayer } from "./mlp";
import type { T5EncoderConfig } from "./types";

/** T5 self-attention sublayer with pre-normalization. */
export class T5LayerSelfAttention extends Module {
  attention: T5Attention;
  layerNorm: RMSNorm;

  constructor(config: T5EncoderConfig, hasRelativeAttentionBias: boolean) {
    super();
    this.attention = new T5Attention(config, hasRelativeAttentionBias);
    this.layerNorm = new RMSNorm(config.dModel, config.layerNormEps);
  }

  forward(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    return this.run(hiddenStates, positionBias);
  }

  run(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    using normalized = this.layerNorm.forward(hiddenStates);
    using attended = this.attention.run(normalized, positionBias);
    return add(hiddenStates, attended);
  }
}

/** T5 feed-forward sublayer with pre-normalization. */
export class T5LayerFeedForward extends Module {
  dense: T5DenseActivationLayer;
  layerNorm: RMSNorm;

  constructor(config: T5EncoderConfig) {
    super();
    this.dense = new T5DenseActivationLayer(config);
    this.layerNorm = new RMSNorm(config.dModel, config.layerNormEps);
  }

  forward(hiddenStates: MxArray): MxArray {
    using normalized = this.layerNorm.forward(hiddenStates);
    using denseOutput = this.dense.forward(normalized);
    return add(hiddenStates, denseOutput);
  }
}

/** Transformer encoder block used by T5EncoderModel. */
export class T5EncoderBlock extends Module {
  selfAttention: T5LayerSelfAttention;
  feedForward: T5LayerFeedForward;

  constructor(config: T5EncoderConfig, hasRelativeAttentionBias: boolean) {
    super();
    this.selfAttention = new T5LayerSelfAttention(config, hasRelativeAttentionBias);
    this.feedForward = new T5LayerFeedForward(config);
  }

  forward(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    return this.run(hiddenStates, positionBias);
  }

  run(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    using attended = this.selfAttention.run(hiddenStates, positionBias);
    return this.feedForward.forward(attended);
  }
}
