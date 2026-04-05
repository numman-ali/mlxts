/**
 * Gemma 4 dense text decoder model and causal LM wrapper.
 * @module
 */

import {
  add,
  compile,
  type DisposableTransform,
  divide,
  formatShape,
  MxArray,
  multiply,
  reshape,
  split,
  squeeze,
  tanh,
} from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import { LayerPatternKVCache } from "../../infrastructure/cache";
import { type AttentionMask, createStepAttentionMask } from "../../infrastructure/masks";
import type { CausalLM, ForwardOptions, TransformerCache } from "../../types";
import { Gemma4TextDecoderBlock } from "./block";
import { Gemma4RMSNorm } from "./norm";
import type { Gemma4LayerType, Gemma4SharedKeyValues, Gemma4TextConfig } from "./types";

function buildSharedKeyValueSourceIndices(
  layerTypes: readonly Gemma4LayerType[],
  numKvSharedLayers: number,
): (number | null)[] {
  const sourceIndices = Array<number | null>(layerTypes.length).fill(null);
  if (numKvSharedLayers === 0) {
    return sourceIndices;
  }

  const firstSharedLayerIndex = layerTypes.length - numKvSharedLayers;
  const latestSourceByType = new Map<Gemma4LayerType, number>();
  for (let layerIndex = 0; layerIndex < firstSharedLayerIndex; layerIndex += 1) {
    const layerType = layerTypes[layerIndex];
    if (layerType !== undefined) {
      latestSourceByType.set(layerType, layerIndex);
    }
  }

  for (let layerIndex = firstSharedLayerIndex; layerIndex < layerTypes.length; layerIndex += 1) {
    const layerType = layerTypes[layerIndex];
    const sourceIndex = layerType === undefined ? undefined : latestSourceByType.get(layerType);
    if (sourceIndex === undefined) {
      throw new Error(
        `Gemma4TextModel: shared KV layer ${layerIndex} has no earlier non-shared ${layerType ?? "unknown"} source.`,
      );
    }
    sourceIndices[layerIndex] = sourceIndex;
  }

  return sourceIndices;
}

function releaseKeyValues(values: (Gemma4SharedKeyValues | null)[]): void {
  for (const value of values) {
    value?.keys.free();
    value?.values.free();
  }
}

function releasePerLayerInputs(values: MxArray[] | null): void {
  if (values === null) {
    return;
  }
  for (const value of values) {
    value.free();
  }
}

/** Decoder backbone shared by Gemma 4 dense text-family checkpoints. */
export class Gemma4TextModel extends Module {
  embedTokens: Embedding;
  layers: Gemma4TextDecoderBlock[];
  norm: Gemma4RMSNorm;
  embedTokensPerLayer: Embedding | null;
  perLayerModelProjection: Linear | null;
  perLayerProjectionNorm: Gemma4RMSNorm | null;
  #embeddingScale: number;
  #perLayerInputScale: number;
  #perLayerProjectionScale: number;
  #hiddenSizePerLayerInput: number;
  #sharedKeyValueSourceIndices: (number | null)[];
  #slidingWindow: number;

  constructor(config: Gemma4TextConfig) {
    super();
    this.#embeddingScale = config.embeddingScale;
    this.#perLayerInputScale = 2 ** -0.5;
    this.#perLayerProjectionScale = config.hiddenSize ** -0.5;
    this.#hiddenSizePerLayerInput = config.hiddenSizePerLayerInput;
    this.#slidingWindow = config.slidingWindow;
    this.#sharedKeyValueSourceIndices = buildSharedKeyValueSourceIndices(
      config.layerTypes,
      config.numKvSharedLayers,
    );
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

  run(inputIds: MxArray, cache?: TransformerCache): MxArray {
    this.assertTokenIds(inputIds, "Gemma4TextModel.forward");
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined) {
      throw new Error("Gemma4TextModel.forward: token ids are missing a sequence axis.");
    }

    using embedded = this.embedTokens.forward(inputIds);
    let hidden = multiply(embedded, this.#embeddingScale);
    const perLayerInputs = this.createPerLayerInputs(inputIds, hidden);
    const attentionMasks = this.createAttentionMasks(
      sequenceLength,
      cache?.offset ?? 0,
      cache !== undefined,
    );

    try {
      const layered = this.runLayers(hidden, cache, perLayerInputs, attentionMasks);
      hidden = layered.hidden;
      releaseKeyValues(layered.keyValues);

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

  private createPerLayerInputs(inputIds: MxArray, inputsEmbeds: MxArray): MxArray[] | null {
    if (
      this.#hiddenSizePerLayerInput === 0 ||
      this.embedTokensPerLayer === null ||
      this.perLayerModelProjection === null ||
      this.perLayerProjectionNorm === null
    ) {
      return null;
    }

    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined) {
      throw new Error(
        `Gemma4TextModel.createPerLayerInputs: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    using embeddedPerLayer = this.embedTokensPerLayer.forward(inputIds);
    using reshapedEmbeddedPerLayer = reshape(embeddedPerLayer, [
      batch,
      sequenceLength,
      this.layers.length,
      this.#hiddenSizePerLayerInput,
    ]);
    using projectedPerLayer = this.perLayerModelProjection.forward(inputsEmbeds);
    using scaledProjectedPerLayer = multiply(projectedPerLayer, this.#perLayerProjectionScale);
    using reshapedProjectedPerLayer = reshape(scaledProjectedPerLayer, [
      batch,
      sequenceLength,
      this.layers.length,
      this.#hiddenSizePerLayerInput,
    ]);
    using normalizedProjectedPerLayer =
      this.perLayerProjectionNorm.forward(reshapedProjectedPerLayer);
    using combinedPerLayer = add(reshapedEmbeddedPerLayer, normalizedProjectedPerLayer);
    using scaledCombinedPerLayer = multiply(combinedPerLayer, this.#perLayerInputScale);
    const chunks = split(scaledCombinedPerLayer, this.layers.length, 2);
    return chunks.map((chunk) => {
      const squeezedChunk = squeeze(chunk, 2);
      chunk.free();
      return squeezedChunk;
    });
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
    const sourceIndex = this.#sharedKeyValueSourceIndices[layerIndex];
    if (sourceIndex === null || sourceIndex === undefined) {
      return undefined;
    }

    const sharedKeyValues = keyValues[sourceIndex] ?? undefined;
    if (sharedKeyValues === undefined) {
      throw new Error(
        `Gemma4TextModel.forward: shared KV source ${sourceIndex} for layer ${layerIndex} is unavailable.`,
      );
    }
    return sharedKeyValues;
  }

  private runLayers(
    hidden: MxArray,
    cache: TransformerCache | undefined,
    perLayerInputs: MxArray[] | null,
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

        const { hidden: nextHidden, keyValues: nextKeyValues } = layer.run(
          currentHidden,
          cache,
          this.sharedKeyValuesForLayer(keyValues, layerIndex),
          perLayerInputs?.[layerIndex],
          attentionMasks[layerIndex],
        );
        currentHidden.free();
        currentHidden = nextHidden;
        keyValues[layerIndex] = nextKeyValues;
      }

      return { hidden: currentHidden, keyValues };
    } catch (error) {
      currentHidden.free();
      releaseKeyValues(keyValues);
      throw error;
    }
  }

  private createAttentionMasks(
    sequenceLength: number,
    pastLength: number,
    trimmedSlidingMasks: boolean,
  ): AttentionMask[] {
    const sharedMasks = new Map<string, AttentionMask>();
    return this.layers.map((layer) => {
      const windowSize =
        layer.selfAttention.layerType === "sliding_attention" ? this.#slidingWindow : undefined;
      const key = windowSize === undefined ? "full" : `sliding:${windowSize}`;
      const existingMask = sharedMasks.get(key);
      if (existingMask !== undefined) {
        return existingMask;
      }
      const attentionMask = createStepAttentionMask(
        sequenceLength,
        pastLength,
        windowSize,
        windowSize !== undefined && trimmedSlidingMasks,
      );
      sharedMasks.set(key, attentionMask);
      return attentionMask;
    });
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
    const finalLogitSoftcapping = this.#finalLogitSoftcapping;
    this.#softcapLogits =
      finalLogitSoftcapping === null
        ? null
        : compile(
            (logits: MxArray) => {
              using scaledLogits = divide(logits, finalLogitSoftcapping);
              using tanhLogits = tanh(scaledLogits);
              return multiply(tanhLogits, finalLogitSoftcapping);
            },
            { shapeless: true },
          );
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

    using hidden = this.model.run(inputIds, options?.cache);
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
