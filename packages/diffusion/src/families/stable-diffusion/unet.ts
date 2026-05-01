/**
 * Stable Diffusion UNet2DConditionModel modules.
 * @module
 */

import { add, concatenate, flatten, formatShape, MxArray, reshape, retainArray } from "@mlxts/core";
import { Conv2d, GroupNorm, Module, silu } from "@mlxts/nn";

import type { StableDiffusionUNetConfig } from "./config";
import {
  StableDiffusionUNetDownBlock2d,
  StableDiffusionUNetMidBlock2d,
  StableDiffusionUNetUpBlock2d,
} from "./unet-blocks";
import {
  StableDiffusionSinusoidalTimesteps,
  StableDiffusionTimestepEmbedding,
} from "./unet-embeddings";

/** Additional text-time conditioning used by SDXL-style UNet checkpoints. */
export type StableDiffusionUNetTextTimeConditioning = {
  textEmbeds: MxArray;
  timeIds: MxArray;
};

/** Options accepted by Stable Diffusion UNet forward calls. */
export type StableDiffusionUNetForwardOptions = {
  textTime?: StableDiffusionUNetTextTimeConditioning;
};

function firstChannel(channels: readonly number[], owner: string): number {
  const channel = channels[0];
  if (channel === undefined) {
    throw new Error(`${owner}: blockOutChannels must not be empty.`);
  }
  return channel;
}

function lastChannel(channels: readonly number[], owner: string): number {
  const channel = channels[channels.length - 1];
  if (channel === undefined) {
    throw new Error(`${owner}: blockOutChannels must not be empty.`);
  }
  return channel;
}

function blockChannel(channels: readonly number[], index: number, owner: string): number {
  const channel = channels[index];
  if (channel === undefined) {
    throw new Error(`${owner}: missing block channel at index ${index}.`);
  }
  return channel;
}

function validateUNetConfig(config: StableDiffusionUNetConfig): void {
  const blockCount = config.blockOutChannels.length;
  if (blockCount === 0) {
    throw new Error("StableDiffusionUNet2DConditionModel: blockOutChannels must not be empty.");
  }
  if (config.downBlockTypes.length !== blockCount || config.upBlockTypes.length !== blockCount) {
    throw new Error(
      "StableDiffusionUNet2DConditionModel: block types must match blockOutChannels.",
    );
  }
  if (
    config.layersPerBlock.length !== blockCount ||
    config.transformerLayersPerBlock.length !== blockCount ||
    config.numAttentionHeads.length !== blockCount ||
    config.crossAttentionDim.length !== blockCount
  ) {
    throw new Error(
      "StableDiffusionUNet2DConditionModel: repeated block fields must match blockOutChannels.",
    );
  }
  for (let index = 0; index < blockCount; index += 1) {
    const channels = blockChannel(
      config.blockOutChannels,
      index,
      "StableDiffusionUNet2DConditionModel",
    );
    const heads = blockChannel(
      config.numAttentionHeads,
      index,
      "StableDiffusionUNet2DConditionModel",
    );
    if (channels % heads !== 0) {
      throw new Error(
        "StableDiffusionUNet2DConditionModel: numAttentionHeads must divide block channels.",
      );
    }
  }
  if (config.upcastAttention) {
    throw new Error("StableDiffusionUNet2DConditionModel: upcastAttention is not supported yet.");
  }
}

function checkedDownBlock(
  blocks: readonly StableDiffusionUNetDownBlock2d[],
  index: number,
): StableDiffusionUNetDownBlock2d {
  const block = blocks[index];
  if (block === undefined) {
    throw new Error(`StableDiffusionUNet2DConditionModel.forward: missing down block ${index}.`);
  }
  return block;
}

function checkedUpBlock(
  blocks: readonly StableDiffusionUNetUpBlock2d[],
  index: number,
): StableDiffusionUNetUpBlock2d {
  const block = blocks[index];
  if (block === undefined) {
    throw new Error(`StableDiffusionUNet2DConditionModel.forward: missing up block ${index}.`);
  }
  return block;
}

function freeAll(values: readonly MxArray[]): void {
  for (const value of values) {
    value.free();
  }
}

function makeDownBlocks(
  config: StableDiffusionUNetConfig,
  timeEmbedDims: number,
): StableDiffusionUNetDownBlock2d[] {
  const first = firstChannel(config.blockOutChannels, "StableDiffusionUNet2DConditionModel");
  return config.blockOutChannels.map((outChannels, index) => {
    const inChannels =
      index === 0
        ? first
        : blockChannel(config.blockOutChannels, index - 1, "StableDiffusionUNet2DConditionModel");
    return new StableDiffusionUNetDownBlock2d({
      inChannels,
      outChannels,
      timeEmbedDims,
      layers: blockChannel(config.layersPerBlock, index, "StableDiffusionUNet2DConditionModel"),
      transformerLayers: blockChannel(
        config.transformerLayersPerBlock,
        index,
        "StableDiffusionUNet2DConditionModel",
      ),
      numHeads: blockChannel(
        config.numAttentionHeads,
        index,
        "StableDiffusionUNet2DConditionModel",
      ),
      crossAttentionDims: blockChannel(
        config.crossAttentionDim,
        index,
        "StableDiffusionUNet2DConditionModel",
      ),
      normGroups: config.normNumGroups,
      normEps: config.normEps,
      addDownsample: index < config.blockOutChannels.length - 1,
      addCrossAttention: config.downBlockTypes[index] === "CrossAttnDownBlock2D",
      useLinearProjection: config.useLinearProjection,
    });
  });
}

function makeUpBlocks(
  config: StableDiffusionUNetConfig,
  timeEmbedDims: number,
): StableDiffusionUNetUpBlock2d[] {
  const reversedChannels = [...config.blockOutChannels].reverse();
  return config.upBlockTypes.map((blockType, index) => {
    const outputChannels = blockChannel(
      reversedChannels,
      index,
      "StableDiffusionUNet2DConditionModel",
    );
    const inputChannels = blockChannel(
      reversedChannels,
      Math.min(index + 1, reversedChannels.length - 1),
      "StableDiffusionUNet2DConditionModel",
    );
    const previousOutputChannels =
      index === 0
        ? firstChannel(reversedChannels, "StableDiffusionUNet2DConditionModel")
        : blockChannel(reversedChannels, index - 1, "StableDiffusionUNet2DConditionModel");
    const reverseIndex = config.upBlockTypes.length - index - 1;
    return new StableDiffusionUNetUpBlock2d({
      inChannels: inputChannels,
      outChannels: outputChannels,
      previousOutputChannels,
      timeEmbedDims,
      layers:
        blockChannel(config.layersPerBlock, reverseIndex, "StableDiffusionUNet2DConditionModel") +
        1,
      transformerLayers: blockChannel(
        config.transformerLayersPerBlock,
        reverseIndex,
        "StableDiffusionUNet2DConditionModel",
      ),
      numHeads: blockChannel(
        config.numAttentionHeads,
        reverseIndex,
        "StableDiffusionUNet2DConditionModel",
      ),
      crossAttentionDims: blockChannel(
        config.crossAttentionDim,
        reverseIndex,
        "StableDiffusionUNet2DConditionModel",
      ),
      normGroups: config.normNumGroups,
      normEps: config.normEps,
      addUpsample: index < config.upBlockTypes.length - 1,
      addCrossAttention: blockType === "CrossAttnUpBlock2D",
      useLinearProjection: config.useLinearProjection,
    });
  });
}

/** Stable Diffusion conditional UNet denoiser over NHWC latent tensors. */
export class StableDiffusionUNet2DConditionModel extends Module {
  convIn: Conv2d;
  timeEmbedding: StableDiffusionTimestepEmbedding;
  addEmbedding: StableDiffusionTimestepEmbedding | null;
  downBlocks: StableDiffusionUNetDownBlock2d[];
  midBlock: StableDiffusionUNetMidBlock2d;
  upBlocks: StableDiffusionUNetUpBlock2d[];
  convNormOut: GroupNorm;
  convOut: Conv2d;
  #timesteps: StableDiffusionSinusoidalTimesteps;
  #addTimeProjection: StableDiffusionSinusoidalTimesteps | null;
  #additionEmbedType: StableDiffusionUNetConfig["additionEmbedType"];
  #timeEmbedDims: number;

  constructor(config: StableDiffusionUNetConfig) {
    super();
    validateUNetConfig(config);
    const first = firstChannel(config.blockOutChannels, "StableDiffusionUNet2DConditionModel");
    const final = lastChannel(config.blockOutChannels, "StableDiffusionUNet2DConditionModel");
    this.#timeEmbedDims = first * 4;
    this.convIn = new Conv2d(
      config.inChannels,
      first,
      config.convInKernel,
      1,
      Math.floor((config.convInKernel - 1) / 2),
    );
    this.#timesteps = new StableDiffusionSinusoidalTimesteps(
      first,
      config.flipSinToCos,
      config.freqShift,
    );
    this.timeEmbedding = new StableDiffusionTimestepEmbedding(first, this.#timeEmbedDims);
    this.#additionEmbedType = config.additionEmbedType;
    this.#addTimeProjection =
      config.additionEmbedType === "text_time" && config.additionTimeEmbedDim !== null
        ? new StableDiffusionSinusoidalTimesteps(
            config.additionTimeEmbedDim,
            config.flipSinToCos,
            config.freqShift,
          )
        : null;
    this.addEmbedding =
      config.additionEmbedType === "text_time" && config.projectionClassEmbeddingsInputDim !== null
        ? new StableDiffusionTimestepEmbedding(
            config.projectionClassEmbeddingsInputDim,
            this.#timeEmbedDims,
          )
        : null;
    this.downBlocks = makeDownBlocks(config, this.#timeEmbedDims);
    this.midBlock = new StableDiffusionUNetMidBlock2d(
      final,
      this.#timeEmbedDims,
      blockChannel(
        config.transformerLayersPerBlock,
        config.transformerLayersPerBlock.length - 1,
        "StableDiffusionUNet2DConditionModel",
      ),
      blockChannel(
        config.numAttentionHeads,
        config.numAttentionHeads.length - 1,
        "StableDiffusionUNet2DConditionModel",
      ),
      blockChannel(
        config.crossAttentionDim,
        config.crossAttentionDim.length - 1,
        "StableDiffusionUNet2DConditionModel",
      ),
      config.normNumGroups,
      config.normEps,
      config.useLinearProjection,
    );
    this.upBlocks = makeUpBlocks(config, this.#timeEmbedDims);
    this.convNormOut = new GroupNorm(config.normNumGroups, first, config.normEps);
    this.convOut = new Conv2d(
      first,
      config.outChannels,
      config.convOutKernel,
      1,
      Math.floor((config.convOutKernel - 1) / 2),
    );
  }

  /** Run the conditional UNet denoiser over NHWC latent tensors. */
  forward(
    x: MxArray,
    timestep: number | MxArray,
    encoderHiddenStates: MxArray,
    options?: StableDiffusionUNetForwardOptions,
  ): MxArray;
  forward(...args: MxArray[]): MxArray;
  forward(
    x: MxArray,
    timestep: number | MxArray,
    encoderHiddenStates: MxArray,
    optionsOrTensor: StableDiffusionUNetForwardOptions | MxArray = {},
  ): MxArray {
    const options = optionsOrTensor instanceof MxArray ? {} : optionsOrTensor;
    const [batch] = this.#assertInputs(x, encoderHiddenStates);
    using timeFeatures = this.#timesteps.forward(timestep, batch, x.dtype);
    let timeEmbedding = this.timeEmbedding.forward(timeFeatures);
    try {
      if (this.#additionEmbedType === "text_time") {
        const augmented = this.#textTimeEmbedding(options.textTime, batch, x.dtype);
        using ownedTimeEmbedding = timeEmbedding;
        using ownedAugmented = augmented;
        timeEmbedding = add(ownedTimeEmbedding, ownedAugmented);
      } else if (options.textTime !== undefined) {
        throw new Error(
          "StableDiffusionUNet2DConditionModel.forward: text-time conditioning is not configured.",
        );
      }
      return this.#forwardWithTimeEmbedding(x, timeEmbedding, encoderHiddenStates);
    } finally {
      timeEmbedding.free();
    }
  }

  #forwardWithTimeEmbedding(
    x: MxArray,
    timeEmbedding: MxArray,
    encoderHiddenStates: MxArray,
  ): MxArray {
    const residuals: MxArray[] = [];
    let hidden = this.convIn.forward(x);
    try {
      residuals.push(retainArray(hidden));
      for (let index = 0; index < this.downBlocks.length; index += 1) {
        const block = checkedDownBlock(this.downBlocks, index);
        const result = block.run(hidden, timeEmbedding, encoderHiddenStates);
        hidden.free();
        hidden = result.hidden;
        residuals.push(...result.residuals);
      }
      const mid = this.midBlock.forward(hidden, timeEmbedding, encoderHiddenStates);
      hidden.free();
      hidden = mid;
      for (let index = 0; index < this.upBlocks.length; index += 1) {
        const block = checkedUpBlock(this.upBlocks, index);
        const nextHidden = block.run(hidden, residuals, timeEmbedding, encoderHiddenStates);
        hidden.free();
        hidden = nextHidden;
      }
      using normalized = this.convNormOut.forward(hidden);
      using activated = silu(normalized);
      const output = this.convOut.forward(activated);
      hidden.free();
      return output;
    } catch (error) {
      hidden.free();
      throw error;
    } finally {
      freeAll(residuals);
    }
  }

  #textTimeEmbedding(
    textTime: StableDiffusionUNetTextTimeConditioning | undefined,
    batch: number,
    dtype: MxArray["dtype"],
  ): MxArray {
    if (textTime === undefined || this.#addTimeProjection === null || this.addEmbedding === null) {
      throw new Error(
        "StableDiffusionUNet2DConditionModel.forward: text-time conditioning is required.",
      );
    }
    if (textTime.textEmbeds.shape.length !== 2 || textTime.textEmbeds.shape[0] !== batch) {
      throw new Error(
        `StableDiffusionUNet2DConditionModel.forward: expected textEmbeds batch ${batch}, got ${formatShape(
          textTime.textEmbeds.shape,
        )}.`,
      );
    }
    if (textTime.timeIds.shape.length < 1 || textTime.timeIds.shape[0] !== batch) {
      throw new Error(
        `StableDiffusionUNet2DConditionModel.forward: expected timeIds batch ${batch}, got ${formatShape(
          textTime.timeIds.shape,
        )}.`,
      );
    }
    const perBatchIds = textTime.timeIds.size / batch;
    using flatTimeIds = flatten(textTime.timeIds);
    using timeFeatures = this.#addTimeProjection.forward(flatTimeIds, batch * perBatchIds, dtype);
    using timeEmbeds = reshape(timeFeatures, [batch, perBatchIds * (timeFeatures.shape[1] ?? 0)]);
    using conditioning = concatenate([textTime.textEmbeds, timeEmbeds], -1);
    return this.addEmbedding.forward(conditioning);
  }

  #assertInputs(
    x: MxArray,
    encoderHiddenStates: MxArray,
  ): readonly [number, number, number, number] {
    const [batch, height, width, channels] = x.shape;
    if (
      x.shape.length !== 4 ||
      batch === undefined ||
      height === undefined ||
      width === undefined ||
      channels === undefined
    ) {
      throw new Error(
        `StableDiffusionUNet2DConditionModel.forward: expected NHWC input, got ${formatShape(x.shape)}.`,
      );
    }
    if (encoderHiddenStates.shape.length !== 3 || encoderHiddenStates.shape[0] !== batch) {
      throw new Error(
        `StableDiffusionUNet2DConditionModel.forward: expected encoder states batch ${batch}, got ${formatShape(
          encoderHiddenStates.shape,
        )}.`,
      );
    }
    return [batch, height, width, channels];
  }

  override [Symbol.dispose](): void {
    this.#timesteps[Symbol.dispose]();
    this.#addTimeProjection?.[Symbol.dispose]();
    super[Symbol.dispose]();
  }
}
