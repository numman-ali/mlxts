/**
 * Gemma 4 dense text decoder model and causal LM wrapper.
 * @module
 */

import { type DisposableTransform, formatShape, MxArray, multiply } from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import { LayerPatternKVCache } from "../../infrastructure/cache";
import { retainInputEmbeddings } from "../../infrastructure/input-embeddings";
import type { AttentionMask } from "../../infrastructure/masks";
import type { CausalLM, ForwardOptions, TransformerCache } from "../../types";
import { Gemma4TextDecoderBlock } from "./block";
import { Gemma4RMSNorm } from "./norm";
import {
  buildSharedKeyValuePlan,
  createAttentionMasks,
  createLogitSoftcap,
  createPerLayerInputs,
  releasePerLayerInputs,
  releaseSharedKeyValues,
  resolvePerLayerInput,
  storeLayerKeyValues,
  takeSharedKeyValuesForLayer,
} from "./runtime/model";
import type { Gemma4SharedKeyValues, Gemma4TextConfig } from "./types";

/** Decoder backbone shared by Gemma 4 dense text-family checkpoints. */
export class Gemma4TextModel extends Module {
  embedTokens: Embedding;
  layers: Gemma4TextDecoderBlock[];
  norm: Gemma4RMSNorm;
  embedTokensPerLayer: Embedding | null;
  perLayerModelProjection: Linear | null;
  perLayerProjectionNorm: Gemma4RMSNorm | null;
  #embeddingScale: number;
  #perLayerEmbeddingScale: number;
  #perLayerInputScale: number;
  #perLayerProjectionScale: number;
  #hiddenSizePerLayerInput: number;
  #hiddenSize: number;
  #sharedKeyValueSourceIndices: (number | null)[];
  #retainedSharedKeyValueSources: boolean[];
  #slidingWindow: number;

  constructor(config: Gemma4TextConfig) {
    super();
    this.#embeddingScale = config.embeddingScale;
    this.#perLayerEmbeddingScale = config.hiddenSizePerLayerInput ** 0.5;
    this.#perLayerInputScale = 2 ** -0.5;
    this.#perLayerProjectionScale = config.hiddenSize ** -0.5;
    this.#hiddenSizePerLayerInput = config.hiddenSizePerLayerInput;
    this.#hiddenSize = config.hiddenSize;
    this.#slidingWindow = config.slidingWindow;
    const sharedKeyValuePlan = buildSharedKeyValuePlan(config.layerTypes, config.numKvSharedLayers);
    this.#sharedKeyValueSourceIndices = sharedKeyValuePlan.sourceIndices;
    this.#retainedSharedKeyValueSources = sharedKeyValuePlan.retainedSourceFlags;
    this.embedTokens = new Embedding(config.vocabSize, config.hiddenSize);
    this.layers = Array.from(
      { length: config.numHiddenLayers },
      (_, layerIndex) => new Gemma4TextDecoderBlock(config, layerIndex),
    );
    this.norm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps);
    if (config.hiddenSizePerLayerInput > 0) {
      this.embedTokensPerLayer = new Embedding(
        config.vocabSizePerLayerInput,
        config.numHiddenLayers * config.hiddenSizePerLayerInput,
      );
      this.perLayerModelProjection = new Linear(
        config.hiddenSize,
        config.numHiddenLayers * config.hiddenSizePerLayerInput,
        false,
      );
      this.perLayerProjectionNorm = new Gemma4RMSNorm(
        config.hiddenSizePerLayerInput,
        config.rmsNormEps,
      );
    } else {
      this.embedTokensPerLayer = null;
      this.perLayerModelProjection = null;
      this.perLayerProjectionNorm = null;
    }
  }

  forward(inputIds: MxArray): MxArray {
    return this.run(inputIds);
  }

  run(inputIds: MxArray, cache?: TransformerCache, inputEmbeddings?: MxArray): MxArray {
    this.assertTokenIds(inputIds, "Gemma4TextModel.forward");
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined) {
      throw new Error("Gemma4TextModel.forward: token ids are missing a sequence axis.");
    }

    using embedded =
      retainInputEmbeddings(
        inputIds,
        inputEmbeddings,
        this.#hiddenSize,
        "Gemma4TextModel.forward",
      ) ?? this.embedTokens.forward(inputIds);
    let hidden = multiply(embedded, this.#embeddingScale);
    const perLayerInputs = createPerLayerInputs(inputIds, hidden, this.perLayerInputModules(), {
      layerCount: this.layers.length,
      hiddenSizePerLayerInput: this.#hiddenSizePerLayerInput,
      embeddingScale: this.#perLayerEmbeddingScale,
      inputScale: this.#perLayerInputScale,
      projectionScale: this.#perLayerProjectionScale,
    });
    const attentionMasks = createAttentionMasks(
      this.layers,
      sequenceLength,
      cache?.offset ?? 0,
      this.#slidingWindow,
      cache !== undefined,
    );

    try {
      const layered = this.runLayers(hidden, cache, perLayerInputs, attentionMasks);
      hidden = layered.hidden;
      releaseSharedKeyValues(layered.keyValues);

      const normalized = this.norm.forward(hidden);
      hidden.free();
      hidden = normalized;
      return hidden;
    } catch (error) {
      hidden.free();
      throw error;
    } finally {
      releasePerLayerInputs(perLayerInputs);
      const releasedMasks = new Set<MxArray>();
      for (const attentionMask of attentionMasks) {
        if (attentionMask instanceof MxArray && !releasedMasks.has(attentionMask)) {
          releasedMasks.add(attentionMask);
          attentionMask.free();
        }
      }
    }
  }

  private assertTokenIds(inputIds: MxArray, context: string): void {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(`${context}: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`);
    }
  }

  private sharedKeyValuesForLayer(
    keyValues: (Gemma4SharedKeyValues | null)[],
    layerIndex: number,
  ): Gemma4SharedKeyValues | undefined {
    return takeSharedKeyValuesForLayer(this.#sharedKeyValueSourceIndices, keyValues, layerIndex);
  }

  private resolvePerLayerInput(perLayerInputs: MxArray | null, layerIndex: number): MxArray | null {
    return resolvePerLayerInput(
      perLayerInputs,
      layerIndex,
      this.#hiddenSizePerLayerInput,
      "Gemma4TextModel.forward",
    );
  }

  private storeLayerKeyValues(
    keyValues: (Gemma4SharedKeyValues | null)[],
    layerIndex: number,
    nextKeyValues: Gemma4SharedKeyValues | null,
  ): void {
    storeLayerKeyValues(this.#retainedSharedKeyValueSources, keyValues, layerIndex, nextKeyValues);
  }

  private runLayers(
    hidden: MxArray,
    cache: TransformerCache | undefined,
    perLayerInputs: MxArray | null,
    attentionMasks: AttentionMask[],
  ): { hidden: MxArray; keyValues: (Gemma4SharedKeyValues | null)[] } {
    let currentHidden = hidden;
    const keyValues = Array<Gemma4SharedKeyValues | null>(this.layers.length).fill(null);

    try {
      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex += 1) {
        const layer = this.layers[layerIndex];
        if (layer === undefined) {
          continue;
        }

        const currentPerLayerInput = this.resolvePerLayerInput(perLayerInputs, layerIndex);
        try {
          const { hidden: nextHidden, keyValues: nextKeyValues } = layer.run(
            currentHidden,
            cache,
            this.sharedKeyValuesForLayer(keyValues, layerIndex),
            currentPerLayerInput ?? undefined,
            attentionMasks[layerIndex],
          );
          currentHidden.free();
          currentHidden = nextHidden;
          this.storeLayerKeyValues(keyValues, layerIndex, nextKeyValues);
        } finally {
          currentPerLayerInput?.free();
        }
      }

      return { hidden: currentHidden, keyValues };
    } catch (error) {
      currentHidden.free();
      releaseSharedKeyValues(keyValues);
      throw error;
    }
  }

  private perLayerInputModules() {
    if (
      this.embedTokensPerLayer === null ||
      this.perLayerModelProjection === null ||
      this.perLayerProjectionNorm === null
    ) {
      return null;
    }

    return {
      embedTokensPerLayer: this.embedTokensPerLayer,
      perLayerModelProjection: this.perLayerModelProjection,
      perLayerProjectionNorm: this.perLayerProjectionNorm,
    };
  }
}

/** Causal LM wrapper for Gemma 4 dense text-family checkpoints. */
export class Gemma4TextCausalLM extends Module implements CausalLM {
  model: Gemma4TextModel;
  lmHead: Linear | null;
  readonly family = "gemma" as const;
  readonly config: Gemma4TextConfig;
  #layerWindowSizes: (number | undefined)[];
  #finalLogitSoftcapping: number | null;
  #softcapLogits: DisposableTransform<(logits: MxArray) => MxArray> | null;

  constructor(config: Gemma4TextConfig) {
    super();
    this.config = config;
    this.#finalLogitSoftcapping = config.finalLogitSoftcapping;
    this.#softcapLogits = createLogitSoftcap(this.#finalLogitSoftcapping);
    this.model = new Gemma4TextModel(config);
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
        `Gemma4TextCausalLM.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    const options =
      optionsOrTensor instanceof Object && !(optionsOrTensor instanceof MxArray)
        ? optionsOrTensor
        : undefined;

    using hidden = this.model.run(inputIds, options?.cache, options?.inputEmbeddings);
    let logits =
      this.lmHead === null ? this.model.embedTokens.asLinear(hidden) : this.lmHead.forward(hidden);

    try {
      if (this.#softcapLogits !== null) {
        const softcappedLogits = this.#softcapLogits(logits);
        logits.free();
        logits = softcappedLogits;
      }
      options?.cache?.advance(sequenceLength);
      return logits;
    } catch (error) {
      logits.free();
      throw error;
    }
  }

  override [Symbol.dispose](): void {
    this.#softcapLogits?.[Symbol.dispose]();
    super[Symbol.dispose]();
  }
}
