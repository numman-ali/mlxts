/**
 * Qwen 3.5 multimodal wrapper and prompt preparation helpers.
 * @module
 */

import {
  array,
  asType,
  formatShape,
  MxArray,
  maskedScatter,
  reshape,
  retainArray,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import { expectSingleTransformerCache } from "../../infrastructure/cache";
import {
  retainInputEmbeddings,
  retainInputPositionIds,
} from "../../infrastructure/input-embeddings";
import type { CausalLM, ForwardOptions, PreparedPrompt, TransformerCache } from "../../types";
import { Qwen3_5TextCache } from "./cache";
import {
  buildPositionIds,
  countImageTokens,
  countQwen3_5ImageTokens,
  createImageMask,
  createQwen3_5MmTokenTypeIds,
  expandQwen3_5ImageTokens,
  gridThwList,
  ropeDeltas,
} from "./conditional-support";
import { Qwen3_5TextModel } from "./model";
import { createShiftedQwen3_5PositionIds } from "./rotary";
import type { Qwen3_5Config } from "./types";
import { Qwen3_5VisionModel } from "./vision";

export { countQwen3_5ImageTokens, createQwen3_5MmTokenTypeIds, expandQwen3_5ImageTokens };

/** Prepare a Qwen 3.5 multimodal prompt against a loaded Qwen 3.5 conditional checkpoint. */
export function prepareQwen3_5ImagePrompt(
  model: CausalLM,
  tokenIds: readonly number[],
  pixelValues: MxArray,
  imageGridThw: MxArray,
  mmTokenTypeIds?: readonly number[],
): PreparedPrompt {
  if (!(model instanceof Qwen3_5ForConditionalGeneration)) {
    throw new Error(
      `prepareQwen3_5ImagePrompt: expected a Qwen 3.5 conditional checkpoint, got model_type "${model.config.modelType}".`,
    );
  }
  return model.prepareImagePrompt(tokenIds, pixelValues, imageGridThw, mmTokenTypeIds);
}

function forwardOptions(optionsOrTensor?: ForwardOptions | MxArray): ForwardOptions | undefined {
  return optionsOrTensor instanceof Object && !(optionsOrTensor instanceof MxArray)
    ? optionsOrTensor
    : undefined;
}

/** Shared multimodal wrapper model without the LM head. */
export class Qwen3_5ConditionalModel extends Module {
  visual: Qwen3_5VisionModel;
  languageModel: Qwen3_5TextModel;

  constructor(config: Qwen3_5Config) {
    super();
    this.visual = new Qwen3_5VisionModel(config.visionConfig);
    this.languageModel = new Qwen3_5TextModel(config.textConfig);
  }

  forward(inputIds: MxArray): MxArray {
    return this.languageModel.run(inputIds);
  }

  run(
    inputIds: MxArray,
    cache?: TransformerCache,
    inputEmbeddings?: MxArray,
    positionIds?: MxArray,
  ): MxArray {
    return this.languageModel.run(inputIds, cache, inputEmbeddings, positionIds);
  }
}

/** Top-level Qwen 3.5 multimodal CausalLM wrapper. */
export class Qwen3_5ForConditionalGeneration extends Module implements CausalLM {
  model: Qwen3_5ConditionalModel;
  lmHead: Linear | null;
  readonly family = "qwen" as const;
  readonly config: Qwen3_5Config;
  #rawConfig: Qwen3_5Config;
  #ropeDeltas: number[] | null = null;

  constructor(config: Qwen3_5Config) {
    super();
    this.#rawConfig = config;
    this.config = config;
    this.model = new Qwen3_5ConditionalModel(config);
    this.lmHead = config.tieWordEmbeddings
      ? null
      : new Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, false);
  }

  get layerCount(): number {
    return this.model.languageModel.layers.length;
  }

  createCache(): TransformerCache {
    return new Qwen3_5TextCache(this.#rawConfig.textConfig.layerTypes);
  }

  prepareImagePrompt(
    tokenIds: readonly number[],
    pixelValues: MxArray,
    imageGridThw: MxArray,
    mmTokenTypeIds?: readonly number[],
  ): PreparedPrompt {
    const grids = gridThwList(imageGridThw, "Qwen3_5ForConditionalGeneration.prepareImagePrompt");
    const requiredImageTokenCount = countImageTokens(
      grids,
      this.#rawConfig.visionConfig.spatialMergeSize,
    );
    const rawImageTokenCount = tokenIds.filter(
      (tokenId) => tokenId === this.#rawConfig.imageTokenId,
    ).length;
    const preparedTokenIds =
      rawImageTokenCount === requiredImageTokenCount
        ? [...tokenIds]
        : rawImageTokenCount === grids.length
          ? expandQwen3_5ImageTokens(
              tokenIds,
              imageGridThw,
              this.#rawConfig.imageTokenId,
              this.#rawConfig.visionConfig.spatialMergeSize,
            )
          : null;
    if (preparedTokenIds === null) {
      throw new Error(
        "Qwen3_5ForConditionalGeneration.prepareImagePrompt: image placeholder count does not match image_grid_thw.",
      );
    }
    const preparedTokenTypes =
      mmTokenTypeIds === undefined
        ? createQwen3_5MmTokenTypeIds(
            preparedTokenIds,
            this.#rawConfig.imageTokenId,
            this.#rawConfig.videoTokenId,
          )
        : [...mmTokenTypeIds];
    if (preparedTokenTypes.length !== preparedTokenIds.length) {
      throw new Error(
        `Qwen3_5ForConditionalGeneration.prepareImagePrompt: mmTokenTypeIds length ${preparedTokenTypes.length} must match prepared token count ${preparedTokenIds.length}.`,
      );
    }
    if (preparedTokenTypes.some((tokenType) => tokenType === 2)) {
      throw new Error(
        "Qwen3_5ForConditionalGeneration.prepareImagePrompt: video token types are not implemented yet.",
      );
    }

    using inputIds = array([preparedTokenIds], "int32");
    using baseEmbeddings = this.model.languageModel.embedTokens.forward(inputIds);
    using visualEmbeddings = this.model.visual.forward(pixelValues, imageGridThw);
    using castVisualEmbeddings =
      visualEmbeddings.dtype === baseEmbeddings.dtype
        ? retainArray(visualEmbeddings)
        : asType(visualEmbeddings, baseEmbeddings.dtype);
    using mask = createImageMask(
      preparedTokenIds,
      this.#rawConfig.imageTokenId,
      this.#rawConfig.textConfig.hiddenSize,
    );
    using flatVisualEmbeddings = reshape(castVisualEmbeddings, [
      (castVisualEmbeddings.shape[0] ?? 0) * this.#rawConfig.textConfig.hiddenSize,
    ]);
    const inputEmbeddings = maskedScatter(baseEmbeddings, mask, flatVisualEmbeddings);
    const positionIds = buildPositionIds(
      preparedTokenIds,
      preparedTokenTypes,
      grids,
      this.#rawConfig.visionConfig.spatialMergeSize,
    );
    return {
      tokenIds: preparedTokenIds,
      inputEmbeddings,
      positionIds,
    };
  }

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray;
  forward(...args: MxArray[]): MxArray;
  forward(inputIds: MxArray, optionsOrTensor?: ForwardOptions | MxArray): MxArray {
    const sequenceLength = inputIds.shape[1];
    const batchSize = inputIds.shape[0];
    if (batchSize === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `Qwen3_5ForConditionalGeneration.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }

    const options = forwardOptions(optionsOrTensor);
    const cache = expectSingleTransformerCache(
      options?.cache,
      "Qwen3_5ForConditionalGeneration.forward",
    );
    const inputEmbeddings = retainInputEmbeddings(
      inputIds,
      options?.inputEmbeddings,
      this.#rawConfig.textConfig.hiddenSize,
      "Qwen3_5ForConditionalGeneration.forward",
    );
    const positionIds = this.prepareForwardPositionIds(
      inputIds,
      options,
      cache,
      batchSize,
      sequenceLength,
    );

    try {
      using hidden = this.model.run(
        inputIds,
        cache,
        inputEmbeddings ?? undefined,
        positionIds ?? undefined,
      );
      const logits =
        this.lmHead === null
          ? this.model.languageModel.embedTokens.asLinear(hidden)
          : this.lmHead.forward(hidden);
      cache?.advance(sequenceLength);
      return logits;
    } finally {
      inputEmbeddings?.free();
      positionIds?.free();
    }
  }

  private prepareForwardPositionIds(
    inputIds: MxArray,
    options: ForwardOptions | undefined,
    cache: TransformerCache | undefined,
    batchSize: number,
    sequenceLength: number,
  ): MxArray | null {
    const positionIds = retainInputPositionIds(
      inputIds,
      options?.positionIds,
      "Qwen3_5ForConditionalGeneration.forward",
    );
    if (cache === undefined || cache.isEmpty()) {
      this.#ropeDeltas = positionIds === null ? null : ropeDeltas(positionIds, sequenceLength);
      return positionIds;
    }
    if (positionIds !== null || this.#ropeDeltas === null) {
      return positionIds;
    }
    if (this.#ropeDeltas.length !== batchSize) {
      throw new Error(
        `Qwen3_5ForConditionalGeneration.forward: cached rope deltas for batch size ${this.#ropeDeltas.length} cannot be reused for batch size ${batchSize}.`,
      );
    }
    return createShiftedQwen3_5PositionIds(
      this.#ropeDeltas.map((delta) => cache.offset + delta),
      sequenceLength,
    );
  }
}
