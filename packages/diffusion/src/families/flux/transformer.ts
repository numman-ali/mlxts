/**
 * FLUX.1 transformer denoiser.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, concatenate, formatShape, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import { FluxDoubleStreamBlock, FluxLastLayer, FluxSingleStreamBlock } from "./blocks";
import type { FluxTransformerConfig } from "./config";
import { FluxEmbedND, FluxMLPEmbedder, fluxTimestepEmbedding } from "./embeddings";
import type { FluxDenoiser, FluxDenoiserInput } from "./pipeline";
import { assertIds2d, assertSequence3d, checkedModule, sliceAxis } from "./tensor-utils";

/** Diffusers-compatible FLUX.1 `FluxTransformer2DModel` tensor path. */
export class FluxTransformer2DModel extends Module implements FluxDenoiser {
  imageProjection: Linear;
  textProjection: Linear;
  timeEmbedding: FluxMLPEmbedder;
  guidanceEmbedding: FluxMLPEmbedder | null;
  vectorEmbedding: FluxMLPEmbedder;
  positionalEmbedding: FluxEmbedND;
  doubleBlocks: FluxDoubleStreamBlock[];
  singleBlocks: FluxSingleStreamBlock[];
  finalLayer: FluxLastLayer;
  #config: FluxTransformerConfig;

  constructor(config: FluxTransformerConfig) {
    super();
    validateFluxTransformerConfig(config);
    this.#config = config;
    const mlpHiddenSize = Math.floor(config.hiddenSize * config.mlpRatio);
    this.imageProjection = new Linear(config.inChannels, config.hiddenSize);
    this.textProjection = new Linear(config.jointAttentionDim, config.hiddenSize);
    this.timeEmbedding = new FluxMLPEmbedder(256, config.hiddenSize);
    this.guidanceEmbedding = config.guidanceEmbeds
      ? new FluxMLPEmbedder(256, config.hiddenSize)
      : null;
    this.vectorEmbedding = new FluxMLPEmbedder(config.pooledProjectionDim, config.hiddenSize);
    this.positionalEmbedding = new FluxEmbedND(
      config.attentionHeadDim,
      config.ropeTheta,
      config.axesDimsRope,
    );
    this.doubleBlocks = Array.from(
      { length: config.numLayers },
      () =>
        new FluxDoubleStreamBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
          mlpHiddenSize,
          qkvBias: config.qkvBias,
        }),
    );
    this.singleBlocks = Array.from(
      { length: config.numSingleLayers },
      () =>
        new FluxSingleStreamBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
          mlpHiddenSize,
          qkvBias: config.qkvBias,
        }),
    );
    this.finalLayer = new FluxLastLayer(config.hiddenSize, config.outChannels);
  }

  forward(input: FluxDenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run a FLUX denoising prediction over packed latent sequence states. */
  forward(inputOrTensor: FluxDenoiserInput | MxArray): MxArray {
    if (!("hiddenStates" in inputOrTensor)) {
      throw new Error("FluxTransformer2DModel.forward: expected a FluxDenoiserInput object.");
    }
    const input = inputOrTensor;
    const imageShape = assertSequence3d(input.hiddenStates, "FluxTransformer2DModel.forward");
    const textShape = assertSequence3d(input.encoderHiddenStates, "FluxTransformer2DModel.forward");
    if (imageShape.channels !== this.#config.inChannels) {
      throw new Error(
        `FluxTransformer2DModel.forward: hiddenStates channels must be ${this.#config.inChannels}.`,
      );
    }
    if (
      textShape.batch !== imageShape.batch ||
      textShape.channels !== this.#config.jointAttentionDim
    ) {
      throw new Error(
        `FluxTransformer2DModel.forward: encoderHiddenStates must have shape [${imageShape.batch}, length, ${this.#config.jointAttentionDim}], got ${formatShape(
          input.encoderHiddenStates.shape,
        )}.`,
      );
    }
    this.#assertVectorShapes(input, imageShape.batch, imageShape.length, textShape.length);

    using imageStates = this.imageProjection.forward(input.hiddenStates);
    using textStates = this.textProjection.forward(input.encoderHiddenStates);
    using timeEmbedding = fluxTimestepEmbedding(input.timestep, 256, {
      dtype: input.hiddenStates.dtype,
      timeFactor: 1000,
    });
    using timeVector = this.timeEmbedding.forward(timeEmbedding);
    using guidedTimeVector = this.#addGuidance(timeVector, input);
    using pooledVector = this.vectorEmbedding.forward(input.pooledProjections);
    using vector = add(guidedTimeVector, pooledVector);
    using ids = concatenate([input.textIds, input.imageIds], 0);
    using rope = this.positionalEmbedding.embed(ids, input.hiddenStates.dtype);

    let image = retainArray(imageStates);
    let text = retainArray(textStates);
    try {
      for (let index = 0; index < this.doubleBlocks.length; index += 1) {
        const block = checkedModule(
          this.doubleBlocks,
          index,
          "FluxTransformer2DModel.forward doubleBlocks",
        );
        const next = block.run(image, text, vector, rope);
        image.free();
        text.free();
        image = next.image;
        text = next.text;
      }
      using joint = concatenate([text, image], 1);
      let hidden = retainArray(joint);
      try {
        for (let index = 0; index < this.singleBlocks.length; index += 1) {
          const block = checkedModule(
            this.singleBlocks,
            index,
            "FluxTransformer2DModel.forward singleBlocks",
          );
          const nextHidden = block.forward(hidden, vector, rope);
          hidden.free();
          hidden = nextHidden;
        }
        using imageHidden = sliceImageHidden(hidden, text.shape[1] ?? 0);
        return this.finalLayer.forward(imageHidden, vector);
      } finally {
        hidden.free();
      }
    } finally {
      image.free();
      text.free();
    }
  }

  #addGuidance(timeVector: MxArray, input: FluxDenoiserInput): MxArray {
    if (this.guidanceEmbedding === null) {
      if (input.guidance !== undefined) {
        throw new Error(
          "FluxTransformer2DModel.forward: guidance is not supported by this config.",
        );
      }
      return retainArray(timeVector);
    }
    if (input.guidance === undefined) {
      throw new Error("FluxTransformer2DModel.forward: guidance is required by this config.");
    }
    using guidanceEmbedding = fluxTimestepEmbedding(input.guidance, 256, {
      dtype: input.hiddenStates.dtype,
      timeFactor: 1000,
    });
    using guidanceVector = this.guidanceEmbedding.forward(guidanceEmbedding);
    return add(timeVector, guidanceVector);
  }

  #assertVectorShapes(
    input: FluxDenoiserInput,
    batch: number,
    imageLength: number,
    textLength: number,
  ): void {
    const imageIdsShape = assertIds2d(input.imageIds, "FluxTransformer2DModel.forward imageIds");
    const textIdsShape = assertIds2d(input.textIds, "FluxTransformer2DModel.forward textIds");
    if (imageIdsShape.length !== imageLength || textIdsShape.length !== textLength) {
      throw new Error(
        "FluxTransformer2DModel.forward: ids lengths must match image/text sequence lengths.",
      );
    }
    const [pooledBatch, pooledChannels] = input.pooledProjections.shape;
    if (
      input.pooledProjections.shape.length !== 2 ||
      pooledBatch !== batch ||
      pooledChannels !== this.#config.pooledProjectionDim
    ) {
      throw new Error(
        `FluxTransformer2DModel.forward: pooledProjections must have shape [${batch}, ${this.#config.pooledProjectionDim}], got ${formatShape(
          input.pooledProjections.shape,
        )}.`,
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== batch) {
      throw new Error(
        `FluxTransformer2DModel.forward: timestep must have shape [${batch}], got ${formatShape(
          input.timestep.shape,
        )}.`,
      );
    }
    if (
      input.guidance !== undefined &&
      (input.guidance.shape.length !== 1 || input.guidance.shape[0] !== batch)
    ) {
      throw new Error(
        `FluxTransformer2DModel.forward: guidance must have shape [${batch}], got ${formatShape(
          input.guidance.shape,
        )}.`,
      );
    }
  }
}

function validateFluxTransformerConfig(config: FluxTransformerConfig): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "FluxTransformer2DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  const ropeDims = config.axesDimsRope.reduce((sum, dim) => sum + dim, 0);
  if (ropeDims !== config.attentionHeadDim) {
    throw new Error("FluxTransformer2DModel: axesDimsRope sum must equal attentionHeadDim.");
  }
  if (config.mlpRatio <= 0) {
    throw new Error("FluxTransformer2DModel: mlpRatio must be positive.");
  }
}

function sliceImageHidden(hiddenStates: MxArray, textLength: number): MxArray {
  const shape = assertSequence3d(hiddenStates, "sliceImageHidden");
  return sliceAxis(hiddenStates, 1, textLength, shape.length);
}
