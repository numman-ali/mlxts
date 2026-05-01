/**
 * FLUX.2 Klein transformer denoiser.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { concatenate, formatShape, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import {
  Flux2AdaptiveLayerNormContinuous,
  Flux2Modulation,
  Flux2SingleTransformerBlock,
  Flux2TransformerBlock,
} from "./blocks";
import type { Flux2KleinTransformerConfig } from "./config";
import { Flux2PosEmbed, Flux2TimestepGuidanceEmbeddings } from "./embeddings";
import type { Flux2KleinDenoiser, Flux2KleinDenoiserInput } from "./pipeline";
import {
  assertFlux2Ids2d,
  assertFlux2Sequence3d,
  checkedFlux2Module,
  sliceFlux2Axis,
} from "./tensor-utils";

/** Diffusers-compatible FLUX.2 Klein `Flux2Transformer2DModel` tensor path. */
export class Flux2KleinTransformer2DModel extends Module implements Flux2KleinDenoiser {
  posEmbed: Flux2PosEmbed;
  timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings;
  doubleStreamModulationImg: Flux2Modulation;
  doubleStreamModulationTxt: Flux2Modulation;
  singleStreamModulation: Flux2Modulation;
  xEmbedder: Linear;
  contextEmbedder: Linear;
  transformerBlocks: Flux2TransformerBlock[];
  singleTransformerBlocks: Flux2SingleTransformerBlock[];
  normOut: Flux2AdaptiveLayerNormContinuous;
  projOut: Linear;
  #config: Flux2KleinTransformerConfig;

  constructor(config: Flux2KleinTransformerConfig) {
    super();
    validateFlux2KleinTransformerConfig(config);
    this.#config = config;
    const mlpHiddenSize = Math.floor(config.hiddenSize * config.mlpRatio);
    this.posEmbed = new Flux2PosEmbed(
      config.attentionHeadDim,
      config.ropeTheta,
      config.axesDimsRope,
    );
    this.timeGuidanceEmbed = new Flux2TimestepGuidanceEmbeddings(
      config.timestepGuidanceChannels,
      config.hiddenSize,
      config.guidanceEmbeds,
    );
    this.doubleStreamModulationImg = new Flux2Modulation(config.hiddenSize, 2);
    this.doubleStreamModulationTxt = new Flux2Modulation(config.hiddenSize, 2);
    this.singleStreamModulation = new Flux2Modulation(config.hiddenSize, 1);
    this.xEmbedder = new Linear(config.inChannels, config.hiddenSize, false);
    this.contextEmbedder = new Linear(config.jointAttentionDim, config.hiddenSize, false);
    this.transformerBlocks = Array.from(
      { length: config.numLayers },
      () =>
        new Flux2TransformerBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
          mlpHiddenSize,
          eps: config.normEps,
        }),
    );
    this.singleTransformerBlocks = Array.from(
      { length: config.numSingleLayers },
      () =>
        new Flux2SingleTransformerBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
          mlpHiddenSize,
          eps: config.normEps,
        }),
    );
    this.normOut = new Flux2AdaptiveLayerNormContinuous(config.hiddenSize, config.normEps);
    this.projOut = new Linear(
      config.hiddenSize,
      config.patchSize * config.patchSize * config.outChannels,
      false,
    );
  }

  forward(input: Flux2KleinDenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run a FLUX.2 Klein denoising prediction over packed latent sequence states. */
  forward(inputOrTensor: Flux2KleinDenoiserInput | MxArray): MxArray {
    if (!("hiddenStates" in inputOrTensor)) {
      throw new Error(
        "Flux2KleinTransformer2DModel.forward: expected a Flux2KleinDenoiserInput object.",
      );
    }
    const input = inputOrTensor;
    const imageShape = assertFlux2Sequence3d(
      input.hiddenStates,
      "Flux2KleinTransformer2DModel.forward",
    );
    const textShape = assertFlux2Sequence3d(
      input.encoderHiddenStates,
      "Flux2KleinTransformer2DModel.forward",
    );
    if (imageShape.channels !== this.#config.inChannels) {
      throw new Error(
        `Flux2KleinTransformer2DModel.forward: hiddenStates channels must be ${this.#config.inChannels}.`,
      );
    }
    if (
      textShape.batch !== imageShape.batch ||
      textShape.channels !== this.#config.jointAttentionDim
    ) {
      throw new Error(
        `Flux2KleinTransformer2DModel.forward: encoderHiddenStates must have shape [${imageShape.batch}, length, ${this.#config.jointAttentionDim}], got ${formatShape(
          input.encoderHiddenStates.shape,
        )}.`,
      );
    }
    this.#assertInputShapes(input, imageShape.batch, imageShape.length, textShape.length);

    using timestepVector = this.timeGuidanceEmbed.forward(input.timestep, input.guidance);
    using doubleImageModulation = this.doubleStreamModulationImg.forward(timestepVector);
    using doubleTextModulation = this.doubleStreamModulationTxt.forward(timestepVector);
    using singleModulation = this.singleStreamModulation.forward(timestepVector);
    using imageStates = this.xEmbedder.forward(input.hiddenStates);
    using textStates = this.contextEmbedder.forward(input.encoderHiddenStates);
    using ids = concatenate([input.textIds, input.imageIds], 0);
    using rope = this.posEmbed.embed(ids, input.hiddenStates.dtype);

    let image = retainArray(imageStates);
    let text = retainArray(textStates);
    try {
      for (let index = 0; index < this.transformerBlocks.length; index += 1) {
        const block = checkedFlux2Module(
          this.transformerBlocks,
          index,
          "Flux2KleinTransformer2DModel.forward transformerBlocks",
        );
        const next = block.run(image, text, doubleImageModulation, doubleTextModulation, rope);
        image.free();
        text.free();
        image = next.image;
        text = next.text;
      }
      using joint = concatenate([text, image], 1);
      let hidden = retainArray(joint);
      try {
        for (let index = 0; index < this.singleTransformerBlocks.length; index += 1) {
          const block = checkedFlux2Module(
            this.singleTransformerBlocks,
            index,
            "Flux2KleinTransformer2DModel.forward singleTransformerBlocks",
          );
          const nextHidden = block.forward(hidden, singleModulation, rope);
          hidden.free();
          hidden = nextHidden;
        }
        using imageHidden = sliceImageHidden(hidden, text.shape[1] ?? 0);
        using normalized = this.normOut.forward(imageHidden, timestepVector);
        return this.projOut.forward(normalized);
      } finally {
        hidden.free();
      }
    } finally {
      image.free();
      text.free();
    }
  }

  #assertInputShapes(
    input: Flux2KleinDenoiserInput,
    batch: number,
    imageLength: number,
    textLength: number,
  ): void {
    const imageIdsShape = assertFlux2Ids2d(
      input.imageIds,
      "Flux2KleinTransformer2DModel.forward imageIds",
    );
    const textIdsShape = assertFlux2Ids2d(
      input.textIds,
      "Flux2KleinTransformer2DModel.forward textIds",
    );
    if (imageIdsShape.length !== imageLength || textIdsShape.length !== textLength) {
      throw new Error(
        "Flux2KleinTransformer2DModel.forward: ids lengths must match image/text sequence lengths.",
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== batch) {
      throw new Error(
        `Flux2KleinTransformer2DModel.forward: timestep must have shape [${batch}], got ${formatShape(
          input.timestep.shape,
        )}.`,
      );
    }
    if (
      input.guidance !== undefined &&
      (input.guidance.shape.length !== 1 || input.guidance.shape[0] !== batch)
    ) {
      throw new Error(
        `Flux2KleinTransformer2DModel.forward: guidance must have shape [${batch}], got ${formatShape(
          input.guidance.shape,
        )}.`,
      );
    }
  }
}

function validateFlux2KleinTransformerConfig(config: Flux2KleinTransformerConfig): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "Flux2KleinTransformer2DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  const ropeDims = config.axesDimsRope.reduce((sum, dim) => sum + dim, 0);
  if (ropeDims !== config.attentionHeadDim) {
    throw new Error("Flux2KleinTransformer2DModel: axesDimsRope sum must equal attentionHeadDim.");
  }
  if (config.mlpRatio <= 0) {
    throw new Error("Flux2KleinTransformer2DModel: mlpRatio must be positive.");
  }
  if (config.patchSize <= 0) {
    throw new Error("Flux2KleinTransformer2DModel: patchSize must be positive.");
  }
}

function sliceImageHidden(hiddenStates: MxArray, textLength: number): MxArray {
  const shape = assertFlux2Sequence3d(hiddenStates, "sliceImageHidden");
  return sliceFlux2Axis(hiddenStates, 1, textLength, shape.length);
}
