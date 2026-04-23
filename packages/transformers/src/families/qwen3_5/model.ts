/**
 * Qwen 3.5 text decoder model and causal LM wrapper.
 * @module
 */

import { formatShape, MxArray } from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import { retainInputEmbeddings } from "../../infrastructure/input-embeddings";
import { type AttentionMask, createStepAttentionMask } from "../../infrastructure/masks";
import type { CausalLM, DecoderCache, ForwardOptions, TransformerCache } from "../../types";
import { Qwen3_5TextDecoderLayer } from "./block";
import { Qwen3_5TextCache } from "./cache";
import { Qwen3_5RMSNorm } from "./norm";
import type { Qwen3_5TextConfig } from "./types";

function expectQwenCache(cache: DecoderCache | undefined): Qwen3_5TextCache | undefined {
  if (cache === undefined) {
    return undefined;
  }
  if (!(cache instanceof Qwen3_5TextCache)) {
    throw new Error(
      `Qwen3_5TextModel.forward: expected Qwen3_5TextCache, got ${cache.constructor.name}.`,
    );
  }
  return cache;
}

function inputShape(inputIds: MxArray): { batchSize: number; sequenceLength: number } {
  const [batchSize, sequenceLength] = inputIds.shape;
  if (batchSize === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
    throw new Error(
      `Qwen3_5TextModel.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
    );
  }
  return { batchSize, sequenceLength };
}

function runDecoderLayer(
  layer: Qwen3_5TextDecoderLayer,
  hiddenStates: MxArray,
  cache: Qwen3_5TextCache | undefined,
  attentionMask: AttentionMask | undefined,
  positionIds?: MxArray,
): MxArray {
  return layer.run(hiddenStates, cache, attentionMask, positionIds);
}

/** Decoder backbone shared by Qwen 3.5 text checkpoints. */
export class Qwen3_5TextModel extends Module {
  embedTokens: Embedding;
  layers: Qwen3_5TextDecoderLayer[];
  norm: Qwen3_5RMSNorm;
  #hiddenSize: number;
  #layerTypes: ReadonlyArray<Qwen3_5TextConfig["layerTypes"][number]>;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.#hiddenSize = config.hiddenSize;
    this.#layerTypes = [...config.layerTypes];
    this.embedTokens = new Embedding(config.vocabSize, config.hiddenSize);
    this.layers = Array.from(
      { length: config.numHiddenLayers },
      (_, layerIndex) => new Qwen3_5TextDecoderLayer(config, layerIndex),
    );
    this.norm = new Qwen3_5RMSNorm(config.hiddenSize, config.rmsNormEps);
  }

  forward(inputIds: MxArray): MxArray {
    return this.run(inputIds);
  }

  run(
    inputIds: MxArray,
    cache?: DecoderCache,
    inputEmbeddings?: MxArray,
    positionIds?: MxArray,
  ): MxArray {
    const { sequenceLength } = inputShape(inputIds);
    const qwenCache = expectQwenCache(cache);
    using embedded =
      retainInputEmbeddings(
        inputIds,
        inputEmbeddings,
        this.#hiddenSize,
        "Qwen3_5TextModel.forward",
      ) ?? this.embedTokens.forward(inputIds);
    const decoded = this.runDecoderLayers(embedded, qwenCache, positionIds, sequenceLength);
    try {
      return this.norm.forward(decoded);
    } finally {
      if (decoded !== embedded) {
        decoded.free();
      }
    }
  }

  private runDecoderLayers(
    initialHiddenStates: MxArray,
    cache: Qwen3_5TextCache | undefined,
    positionIds: MxArray | undefined,
    sequenceLength: number,
  ): MxArray {
    let hiddenStates = initialHiddenStates;
    const fullAttentionMask = createStepAttentionMask(sequenceLength, cache?.offset ?? 0);
    try {
      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex += 1) {
        const layer = this.layers[layerIndex];
        if (layer === undefined) {
          throw new Error(`Qwen3_5TextModel.run: layer ${layerIndex} is missing.`);
        }
        const nextHiddenStates = runDecoderLayer(
          layer,
          hiddenStates,
          cache,
          this.#layerTypes[layerIndex] === "full_attention" ? fullAttentionMask : undefined,
          positionIds,
        );
        if (hiddenStates !== initialHiddenStates) {
          hiddenStates.free();
        }
        hiddenStates = nextHiddenStates;
      }
      return hiddenStates;
    } catch (error) {
      if (hiddenStates !== initialHiddenStates) {
        hiddenStates.free();
      }
      throw error;
    } finally {
      if (fullAttentionMask instanceof MxArray) {
        fullAttentionMask.free();
      }
    }
  }
}

/** Causal LM wrapper for Qwen 3.5 text checkpoints. */
export class Qwen3_5TextCausalLM extends Module implements CausalLM {
  model: Qwen3_5TextModel;
  lmHead: Linear | null;
  readonly family = "qwen" as const;
  readonly config: Qwen3_5TextConfig;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.config = config;
    this.model = new Qwen3_5TextModel(config);
    this.lmHead = config.tieWordEmbeddings
      ? null
      : new Linear(config.hiddenSize, config.vocabSize, false);
  }

  get layerCount(): number {
    return this.model.layers.length;
  }

  createCache(): TransformerCache {
    return new Qwen3_5TextCache(this.config.layerTypes);
  }

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray;
  forward(...args: MxArray[]): MxArray;
  forward(inputIds: MxArray, optionsOrTensor?: ForwardOptions | MxArray): MxArray {
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `Qwen3_5TextCausalLM.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    const options =
      optionsOrTensor instanceof Object && !(optionsOrTensor instanceof MxArray)
        ? optionsOrTensor
        : undefined;

    using hidden = this.model.run(
      inputIds,
      options?.cache,
      options?.inputEmbeddings,
      options?.positionIds,
    );
    const logits =
      this.lmHead === null ? this.model.embedTokens.asLinear(hidden) : this.lmHead.forward(hidden);
    options?.cache?.advance(sequenceLength);
    return logits;
  }
}
