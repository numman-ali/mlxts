/**
 * Shared LLaMA-like decoder model and causal LM wrapper.
 * @module
 */

import { formatShape, MxArray, multiply } from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import { KVCache, SlidingWindowKVCache } from "../../infrastructure/cache";
import { retainInputEmbeddings } from "../../infrastructure/input-embeddings";
import { createStepAttentionMask } from "../../infrastructure/masks";
import type { CausalLM, ForwardOptions, TransformerCache } from "../../types";
import { LlamaLikeDecoderBlock } from "./block";
import { LlamaLikeNorm } from "./norm";
import type { LlamaLikeConfig } from "./types";

/** Decoder backbone shared by the supported LLaMA-like families. */
export class LlamaLikeModel extends Module {
  embedTokens: Embedding;
  layers: LlamaLikeDecoderBlock[];
  norm: LlamaLikeNorm;
  #embeddingScale: number;
  #hiddenSize: number;

  constructor(config: LlamaLikeConfig) {
    super();
    this.#embeddingScale = config.embeddingScale ?? 1.0;
    this.#hiddenSize = config.hiddenSize;
    this.embedTokens = new Embedding(config.vocabSize, config.hiddenSize);
    this.layers = Array.from(
      { length: config.numHiddenLayers },
      () => new LlamaLikeDecoderBlock(config),
    );
    this.norm = new LlamaLikeNorm(config);
  }

  forward(inputIds: MxArray): MxArray {
    return this.run(inputIds);
  }

  run(inputIds: MxArray, cache?: TransformerCache, inputEmbeddings?: MxArray): MxArray {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `LlamaLikeModel.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    using embedded =
      retainInputEmbeddings(
        inputIds,
        inputEmbeddings,
        this.#hiddenSize,
        "LlamaLikeModel.forward",
      ) ?? this.embedTokens.forward(inputIds);
    let hidden = this.#embeddingScale === 1.0 ? embedded : multiply(embedded, this.#embeddingScale);
    const attentionMask = createStepAttentionMask(sequenceLength, cache?.offset ?? 0, undefined);

    try {
      for (let index = 0; index < this.layers.length; index += 1) {
        const layer = this.layers[index];
        if (layer === undefined) {
          continue;
        }

        const nextHidden = layer.run(hidden, index, cache, attentionMask);
        hidden.free();
        hidden = nextHidden;
      }

      const normalized = this.norm.forward(hidden);
      hidden.free();
      hidden = normalized;
      return hidden;
    } catch (error) {
      hidden.free();
      throw error;
    } finally {
      if (attentionMask instanceof MxArray) {
        attentionMask.free();
      }
    }
  }
}

/** Shared LLaMA-like causal LM used by the supported dense decoder families. */
export class LlamaLikeCausalLM extends Module implements CausalLM {
  model: LlamaLikeModel;
  lmHead: Linear | null;
  readonly family: LlamaLikeConfig["family"];
  readonly config: LlamaLikeConfig;
  #slidingWindow: number | undefined;

  constructor(config: LlamaLikeConfig) {
    super();
    this.family = config.family;
    this.config = config;
    this.#slidingWindow = config.slidingWindow;
    this.model = new LlamaLikeModel(config);
    this.lmHead = config.tieWordEmbeddings
      ? null
      : new Linear(config.hiddenSize, config.vocabSize, false);
  }

  get layerCount(): number {
    return this.model.layers.length;
  }

  createCache(): TransformerCache {
    return this.#slidingWindow === undefined
      ? new KVCache(this.layerCount)
      : new SlidingWindowKVCache(this.layerCount, this.#slidingWindow);
  }

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray;
  forward(...args: MxArray[]): MxArray;
  forward(inputIds: MxArray, optionsOrTensor?: ForwardOptions | MxArray): MxArray {
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `LlamaLikeCausalLM.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    const options =
      optionsOrTensor instanceof Object && !(optionsOrTensor instanceof MxArray)
        ? optionsOrTensor
        : undefined;

    using hidden = this.model.run(inputIds, options?.cache, options?.inputEmbeddings);
    const logits =
      this.lmHead === null ? this.model.embedTokens.asLinear(hidden) : this.lmHead.forward(hidden);
    options?.cache?.advance(sequenceLength);
    return logits;
  }
}
