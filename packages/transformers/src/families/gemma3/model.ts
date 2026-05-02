/**
 * Gemma 3 text decoder model and causal LM wrapper.
 * @module
 */

import { formatShape, MxArray, multiply, retainArray } from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import {
  isManagedBatchKVCache,
  isManagedLayerPatternBatchKVCache,
  isSingleTransformerCache,
  type LayerPatternBatchKVCache,
  LayerPatternKVCache,
} from "../../infrastructure/cache";
import { retainInputEmbeddings } from "../../infrastructure/input-embeddings";
import {
  type AttentionMask,
  createStepAttentionMask,
  createTokenizerCausalAttentionMask,
} from "../../infrastructure/masks";
import type { CausalLM, DecoderCache, ForwardOptions, TransformerCache } from "../../types";
import { Gemma3DecoderBlock } from "./block";
import { Gemma3RMSNorm } from "./norm";
import type { Gemma3TextConfig } from "./types";

export type Gemma3Cache = TransformerCache | LayerPatternBatchKVCache;

/** Runtime options for a Gemma 3 decoder backbone pass. */
export type Gemma3TextModelOptions = {
  cache?: Gemma3Cache;
  inputEmbeddings?: MxArray;
  attentionMask?: MxArray;
  outputHiddenStates?: boolean;
};

/** Retained Gemma 3 decoder backbone outputs for conditioning consumers. */
export type Gemma3TextModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

function expectGemma3Cache(
  cache: DecoderCache | undefined,
  context: string,
): Gemma3Cache | undefined {
  if (cache === undefined) {
    return undefined;
  }
  if (isSingleTransformerCache(cache) || isManagedLayerPatternBatchKVCache(cache)) {
    return cache;
  }
  const cacheName = isManagedBatchKVCache(cache) ? "BatchKVCache" : "batch cache";
  throw new Error(`${context}: ${cacheName} is not supported by Gemma 3.`);
}

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

function expectRunShape(
  inputIds: MxArray,
  options: Gemma3TextModelOptions,
): { sequenceLength: number } {
  const [batch, sequenceLength] = inputIds.shape;
  if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
    throw new Error(
      `Gemma3TextModel.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
    );
  }
  if (options.attentionMask !== undefined && options.cache !== undefined) {
    throw new Error("Gemma3TextModel.forward: attentionMask is only supported without cache.");
  }
  if (
    options.attentionMask !== undefined &&
    (options.attentionMask.shape[0] !== batch || options.attentionMask.shape[1] !== sequenceLength)
  ) {
    throw new Error(
      `Gemma3TextModel.forward: attentionMask must match token ids, got ${formatShape(
        options.attentionMask.shape,
      )}.`,
    );
  }
  return { sequenceLength };
}

function retainHiddenState(hiddenStates: MxArray[] | undefined, hidden: MxArray): void {
  hiddenStates?.push(retainArray(hidden));
}

function gemma3Output(
  lastHiddenState: MxArray,
  hiddenStates: MxArray[] | undefined,
): Gemma3TextModelOutput {
  const output: Gemma3TextModelOutput = { lastHiddenState };
  if (hiddenStates !== undefined) {
    output.hiddenStates = hiddenStates;
  }
  return output;
}

function releaseAttentionMasks(attentionMasks: readonly (AttentionMask | undefined)[]): void {
  const releasedMasks = new Set<MxArray>();
  for (const attentionMask of attentionMasks) {
    if (attentionMask instanceof MxArray && !releasedMasks.has(attentionMask)) {
      releasedMasks.add(attentionMask);
      attentionMask.free();
    }
  }
}

/** Dispose arrays returned by `Gemma3TextModel.runWithHiddenStates`. */
export function disposeGemma3TextModelOutput(output: Gemma3TextModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

/** Decoder backbone shared by Gemma 3 text checkpoints. */
export class Gemma3TextModel extends Module {
  embedTokens: Embedding;
  layers: Gemma3DecoderBlock[];
  norm: Gemma3RMSNorm;
  #embeddingScale: number;
  #slidingWindow: number;
  #hiddenSize: number;

  constructor(config: Gemma3TextConfig) {
    super();
    this.#embeddingScale = config.embeddingScale;
    this.#slidingWindow = config.slidingWindow;
    this.#hiddenSize = config.hiddenSize;
    this.embedTokens = new Embedding(config.vocabSize, config.hiddenSize);
    this.layers = Array.from(
      { length: config.numHiddenLayers },
      (_, layerIndex) => new Gemma3DecoderBlock(config, layerIndex),
    );
    this.norm = new Gemma3RMSNorm(config.hiddenSize, config.rmsNormEps);
  }

  forward(inputIds: MxArray): MxArray {
    return this.run(inputIds);
  }

  run(inputIds: MxArray, cache?: Gemma3Cache, inputEmbeddings?: MxArray): MxArray {
    const options: Gemma3TextModelOptions = {};
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
    options: Gemma3TextModelOptions = {},
  ): Gemma3TextModelOutput {
    return this.runModel(inputIds, options);
  }

  private runModel(inputIds: MxArray, options: Gemma3TextModelOptions): Gemma3TextModelOutput {
    const { sequenceLength } = expectRunShape(inputIds, options);

    using embedded =
      retainInputEmbeddings(
        inputIds,
        options.inputEmbeddings,
        this.#hiddenSize,
        "Gemma3TextModel.forward",
      ) ?? this.embedTokens.forward(inputIds);
    let hidden = multiply(embedded, this.#embeddingScale);
    const hiddenStates: MxArray[] | undefined = options.outputHiddenStates ? [] : undefined;
    const attentionMasks = this.createAttentionMasks(
      sequenceLength,
      options.cache,
      options.attentionMask,
    );
    let lastHiddenState: MxArray | null = null;

    try {
      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex += 1) {
        const layer = this.layers[layerIndex];
        if (layer === undefined) {
          continue;
        }
        retainHiddenState(hiddenStates, hidden);
        const nextHidden = layer.run(hidden, options.cache, attentionMasks[layerIndex]);
        hidden.free();
        hidden = nextHidden;
      }

      lastHiddenState = this.norm.forward(hidden);
      retainHiddenState(hiddenStates, lastHiddenState);
      return gemma3Output(lastHiddenState, hiddenStates);
    } catch (error) {
      lastHiddenState?.free();
      disposeHiddenStates(hiddenStates);
      throw error;
    } finally {
      releaseAttentionMasks(attentionMasks);
      hidden.free();
    }
  }

  private createAttentionMasks(
    sequenceLength: number,
    cache: Gemma3Cache | undefined,
    tokenizerAttentionMask: MxArray | undefined,
  ): (AttentionMask | undefined)[] {
    if (tokenizerAttentionMask !== undefined) {
      const sharedMasks = new Map<string, AttentionMask>();
      return this.layers.map((layer) => {
        const key = layer.isSliding ? `sliding:${this.#slidingWindow}` : "full";
        const existingMask = sharedMasks.get(key);
        if (existingMask !== undefined) {
          return existingMask;
        }
        const attentionMask = createTokenizerCausalAttentionMask(
          tokenizerAttentionMask,
          layer.isSliding ? this.#slidingWindow : undefined,
        );
        sharedMasks.set(key, attentionMask);
        return attentionMask;
      });
    }
    if (isManagedLayerPatternBatchKVCache(cache)) {
      return this.layers.map(() => undefined);
    }

    const pastLength = cache?.offset ?? 0;
    const sharedMasks = new Map<string, AttentionMask>();
    return this.layers.map((layer) => {
      const key = layer.isSliding ? `sliding:${this.#slidingWindow}` : "full";
      const existingMask = sharedMasks.get(key);
      if (existingMask !== undefined) {
        return existingMask;
      }
      const attentionMask = createStepAttentionMask(
        sequenceLength,
        pastLength,
        layer.isSliding ? this.#slidingWindow : undefined,
        layer.isSliding && cache !== undefined,
      );
      sharedMasks.set(key, attentionMask);
      return attentionMask;
    });
  }
}

/** Causal LM wrapper for Gemma 3 text-family checkpoints. */
export class Gemma3TextCausalLM extends Module implements CausalLM {
  model: Gemma3TextModel;
  lmHead: Linear | null;
  readonly family = "gemma" as const;
  readonly config: Gemma3TextConfig;
  #layerWindowSizes: (number | undefined)[];

  constructor(config: Gemma3TextConfig) {
    super();
    this.config = config;
    this.model = new Gemma3TextModel(config);
    this.lmHead = config.tieWordEmbeddings
      ? null
      : new Linear(config.hiddenSize, config.vocabSize, false);
    this.#layerWindowSizes = config.layerTypes.map((layerType) =>
      layerType === "sliding_attention" ? config.slidingWindow : undefined,
    );
  }

  get layerCount(): number {
    return this.model.layers.length;
  }

  createCache(): TransformerCache {
    return new LayerPatternKVCache(this.layerCount, this.#layerWindowSizes);
  }

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray;
  forward(...args: MxArray[]): MxArray;
  forward(inputIds: MxArray, optionsOrTensor?: ForwardOptions | MxArray): MxArray {
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `Gemma3TextCausalLM.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    const options =
      optionsOrTensor instanceof Object && !(optionsOrTensor instanceof MxArray)
        ? optionsOrTensor
        : undefined;

    const cache = expectGemma3Cache(options?.cache, "Gemma3TextCausalLM.forward");
    using hidden = this.model.run(inputIds, cache, options?.inputEmbeddings);
    const logits =
      this.lmHead === null ? this.model.embedTokens.asLinear(hidden) : this.lmHead.forward(hidden);
    cache?.advance(sequenceLength);
    return logits;
  }
}
