/**
 * LLaMA-style decoder block.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import type { AttentionMask } from "../../infrastructure/masks";
import type { DecoderCache } from "../../types";
import { LlamaLikeAttention } from "./attention";
import { LlamaLikeMLP } from "./mlp";
import { LlamaLikeNorm } from "./norm";
import type { ForwardModule, LlamaLikeConfig } from "./types";

/** Decoder block with attention, residual, and MLP stages. */
export class LlamaLikeDecoderBlock extends Module {
  inputLayerNorm: LlamaLikeNorm;
  selfAttention: LlamaLikeAttention;
  postAttentionLayerNorm: LlamaLikeNorm;
  mlp: ForwardModule;

  constructor(config: LlamaLikeConfig) {
    super();
    this.inputLayerNorm = new LlamaLikeNorm(config);
    this.selfAttention = new LlamaLikeAttention(config);
    this.postAttentionLayerNorm = new LlamaLikeNorm(config);
    this.mlp = new LlamaLikeMLP(config);
  }

  forward(x: MxArray): MxArray {
    return this.run(x, 0);
  }

  run(
    x: MxArray,
    layerIndex: number,
    cache?: DecoderCache,
    attentionMask?: AttentionMask,
  ): MxArray {
    using normalizedForAttention = this.inputLayerNorm.forward(x);
    using attentionOutput = this.selfAttention.run(
      normalizedForAttention,
      layerIndex,
      cache,
      attentionMask,
    );
    using residualAfterAttention = add(x, attentionOutput);
    using normalizedForMlp = this.postAttentionLayerNorm.forward(residualAfterAttention);
    using mlpOutput = this.mlp.forward(normalizedForMlp);
    return add(residualAfterAttention, mlpOutput);
  }
}
