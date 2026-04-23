/**
 * Qwen 3.5 text decoder block.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import type { AttentionMask } from "../../infrastructure/masks";
import { Qwen3_5TextAttention } from "./attention";
import type { Qwen3_5TextCache } from "./cache";
import { Qwen3_5GatedDeltaNet } from "./gated-delta";
import { Qwen3_5TextMLP } from "./mlp";
import { Qwen3_5RMSNorm } from "./norm";
import type { Qwen3_5TextConfig } from "./types";

/** One Qwen 3.5 decoder layer with either full or linear attention. */
export class Qwen3_5TextDecoderLayer extends Module {
  selfAttention: Qwen3_5TextAttention | null;
  linearAttention: Qwen3_5GatedDeltaNet | null;
  mlp: Qwen3_5TextMLP;
  inputLayerNorm: Qwen3_5RMSNorm;
  postAttentionLayerNorm: Qwen3_5RMSNorm;
  #layerIndex: number;
  #layerType: Qwen3_5TextConfig["layerTypes"][number];

  constructor(config: Qwen3_5TextConfig, layerIndex: number) {
    super();
    this.#layerIndex = layerIndex;
    this.#layerType = config.layerTypes[layerIndex] ?? "linear_attention";
    this.selfAttention =
      this.#layerType === "full_attention" ? new Qwen3_5TextAttention(config) : null;
    this.linearAttention =
      this.#layerType === "linear_attention" ? new Qwen3_5GatedDeltaNet(config) : null;
    this.mlp = new Qwen3_5TextMLP(config);
    this.inputLayerNorm = new Qwen3_5RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.postAttentionLayerNorm = new Qwen3_5RMSNorm(config.hiddenSize, config.rmsNormEps);
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(
    x: MxArray,
    cache?: Qwen3_5TextCache,
    attentionMask?: AttentionMask,
    positionIds?: MxArray,
  ): MxArray {
    using normalizedInputs = this.inputLayerNorm.forward(x);
    using tokenMixer = this.runTokenMixer(normalizedInputs, cache, attentionMask, positionIds);
    using residual = add(x, tokenMixer);
    using normalizedResidual = this.postAttentionLayerNorm.forward(residual);
    using mlpOutput = this.mlp.forward(normalizedResidual);
    return add(residual, mlpOutput);
  }

  private runTokenMixer(
    normalizedInputs: MxArray,
    cache: Qwen3_5TextCache | undefined,
    attentionMask: AttentionMask | undefined,
    positionIds: MxArray | undefined,
  ): MxArray {
    if (this.selfAttention !== null) {
      return this.selfAttention.run(
        normalizedInputs,
        this.#layerIndex,
        cache,
        attentionMask,
        positionIds,
      );
    }
    if (this.linearAttention !== null) {
      return this.linearAttention.run(normalizedInputs, this.#layerIndex, cache);
    }
    throw new Error(
      `Qwen3_5TextDecoderLayer.run: layer ${this.#layerIndex} is missing a token mixer.`,
    );
  }
}
