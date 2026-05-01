/**
 * Shared LLaMA-like decoder model and causal LM wrapper.
 * @module
 */

import { formatShape, MxArray, multiply, retainArray } from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import {
  isManagedBatchKVCache,
  isSingleTransformerCache,
  KVCache,
  SlidingWindowKVCache,
} from "../../infrastructure/cache";
import { retainInputEmbeddings } from "../../infrastructure/input-embeddings";
import {
  type AttentionMask,
  createLeftPaddedAttentionMask,
  createStepAttentionMask,
} from "../../infrastructure/masks";
import type { CausalLM, DecoderCache, ForwardOptions, TransformerCache } from "../../types";
import { LlamaLikeDecoderBlock } from "./block";
import { LlamaLikeNorm } from "./norm";
import type { LlamaLikeConfig } from "./types";

/** Runtime options for a shared LLaMA-like decoder backbone pass. */
export type LlamaLikeModelOptions = {
  cache?: DecoderCache;
  inputEmbeddings?: MxArray;
  outputHiddenStates?: boolean;
};

/** Retained decoder backbone outputs for encoder-style conditioning consumers. */
export type LlamaLikeModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

function createAttentionMask(
  sequenceLength: number,
  cache: DecoderCache | undefined,
): AttentionMask {
  if (cache === undefined) {
    return createStepAttentionMask(sequenceLength, 0, undefined);
  }
  if (isSingleTransformerCache(cache)) {
    return createStepAttentionMask(sequenceLength, cache.offset, undefined);
  }
  if (!isManagedBatchKVCache(cache)) {
    throw new Error("LlamaLikeModel.forward: unsupported batch cache implementation.");
  }

  using leftPadding = cache.leftPaddingTensor();
  return createLeftPaddedAttentionMask(
    sequenceLength,
    cache.length + sequenceLength,
    cache.length,
    leftPadding,
  );
}

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

/** Dispose arrays returned by `LlamaLikeModel.runWithHiddenStates`. */
export function disposeLlamaLikeModelOutput(output: LlamaLikeModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

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

  run(inputIds: MxArray, cache?: DecoderCache, inputEmbeddings?: MxArray): MxArray {
    const options: LlamaLikeModelOptions = {};
    if (cache !== undefined) {
      options.cache = cache;
    }
    if (inputEmbeddings !== undefined) {
      options.inputEmbeddings = inputEmbeddings;
    }
    const output = this.runModel(inputIds, options);
    disposeHiddenStates(output.hiddenStates);
    return output.lastHiddenState;
  }

  /** Run the decoder backbone and retain hidden states for conditioning paths. */
  runWithHiddenStates(
    inputIds: MxArray,
    options: LlamaLikeModelOptions = {},
  ): LlamaLikeModelOutput {
    return this.runModel(inputIds, options);
  }

  private runModel(inputIds: MxArray, options: LlamaLikeModelOptions): LlamaLikeModelOutput {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `LlamaLikeModel.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    using embedded =
      retainInputEmbeddings(
        inputIds,
        options.inputEmbeddings,
        this.#hiddenSize,
        "LlamaLikeModel.forward",
      ) ?? this.embedTokens.forward(inputIds);
    let hidden = this.#embeddingScale === 1.0 ? embedded : multiply(embedded, this.#embeddingScale);
    const hiddenStates: MxArray[] | undefined = options.outputHiddenStates ? [] : undefined;
    const attentionMask = createAttentionMask(sequenceLength, options.cache);
    let lastHiddenState: MxArray | null = null;

    try {
      for (let index = 0; index < this.layers.length; index += 1) {
        const layer = this.layers[index];
        if (layer === undefined) {
          continue;
        }

        if (hiddenStates !== undefined) {
          hiddenStates.push(retainArray(hidden));
        }
        const nextHidden = layer.run(hidden, index, options.cache, attentionMask);
        hidden.free();
        hidden = nextHidden;
      }

      lastHiddenState = this.norm.forward(hidden);
      if (hiddenStates !== undefined) {
        hiddenStates.push(retainArray(lastHiddenState));
      }
      const output: LlamaLikeModelOutput = { lastHiddenState };
      if (hiddenStates !== undefined) {
        output.hiddenStates = hiddenStates;
      }
      return output;
    } catch (error) {
      lastHiddenState?.free();
      disposeHiddenStates(hiddenStates);
      throw error;
    } finally {
      if (attentionMask instanceof MxArray) {
        attentionMask.free();
      }
      hidden.free();
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

    if (options?.cache !== undefined && isManagedBatchKVCache(options.cache)) {
      if (this.#slidingWindow !== undefined) {
        throw new Error(
          "LlamaLikeCausalLM.forward: BatchKVCache is only supported for full-cache models.",
        );
      }
    }

    using hidden = this.model.run(inputIds, options?.cache, options?.inputEmbeddings);
    const logits =
      this.lmHead === null ? this.model.embedTokens.asLinear(hidden) : this.lmHead.forward(hidden);
    options?.cache?.advance(sequenceLength);
    return logits;
  }
}
