import { formatShape, MxArray, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { StableDiffusion3JointTransformerBlock } from "./blocks";
import type { StableDiffusion3TransformerConfig } from "./config";
import { StableDiffusion3PatchEmbed, StableDiffusion3TimestepTextEmbeddings } from "./embeddings";
import { unpatchifyStableDiffusion3Latents } from "./latents";
import { StableDiffusion3AdaLayerNormContinuous } from "./normalization";
import { assertImage4d, assertSequence3d, checkedModule } from "./tensor-utils";

/** Prepared conditioning tensors consumed by the SD3 transformer denoiser. */
export type StableDiffusion3DenoiserInput = {
  hiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  pooledProjections: MxArray;
  timestep: MxArray;
};

function validateStableDiffusion3TransformerConfig(
  config: StableDiffusion3TransformerConfig,
): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "StableDiffusion3Transformer2DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  if (config.captionProjectionDim !== config.hiddenSize) {
    throw new Error(
      "StableDiffusion3Transformer2DModel: captionProjectionDim must equal hiddenSize.",
    );
  }
  if (config.hiddenSize % 4 !== 0) {
    throw new Error("StableDiffusion3Transformer2DModel: hiddenSize must be divisible by 4.");
  }
}

function dualAttentionLayerSet(config: StableDiffusion3TransformerConfig): ReadonlySet<number> {
  return new Set(config.dualAttentionLayers);
}

/** Diffusers-compatible Stable Diffusion 3 `SD3Transformer2DModel` tensor path. */
export class StableDiffusion3Transformer2DModel extends Module {
  posEmbed: StableDiffusion3PatchEmbed;
  timeTextEmbed: StableDiffusion3TimestepTextEmbeddings;
  contextEmbedder: Linear;
  transformerBlocks: StableDiffusion3JointTransformerBlock[];
  normOut: StableDiffusion3AdaLayerNormContinuous;
  projOut: Linear;
  #config: StableDiffusion3TransformerConfig;

  constructor(config: StableDiffusion3TransformerConfig) {
    super();
    validateStableDiffusion3TransformerConfig(config);
    this.#config = config;
    this.posEmbed = new StableDiffusion3PatchEmbed({
      sampleSize: config.sampleSize,
      patchSize: config.patchSize,
      inChannels: config.inChannels,
      hiddenSize: config.hiddenSize,
      posEmbedMaxSize: config.posEmbedMaxSize,
    });
    this.timeTextEmbed = new StableDiffusion3TimestepTextEmbeddings(
      config.hiddenSize,
      config.pooledProjectionDim,
    );
    this.contextEmbedder = new Linear(config.jointAttentionDim, config.captionProjectionDim);
    const dualLayers = dualAttentionLayerSet(config);
    this.transformerBlocks = Array.from(
      { length: config.numLayers },
      (_unused, index) =>
        new StableDiffusion3JointTransformerBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
          qkNorm: config.qkNorm,
          contextPreOnly: index === config.numLayers - 1,
          useDualAttention: dualLayers.has(index),
        }),
    );
    this.normOut = new StableDiffusion3AdaLayerNormContinuous(config.hiddenSize, config.hiddenSize);
    this.projOut = new Linear(
      config.hiddenSize,
      config.patchSize * config.patchSize * config.outChannels,
    );
  }

  forward(input: StableDiffusion3DenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run an SD3 denoising prediction over NHWC latent tensors. */
  forward(inputOrTensor: StableDiffusion3DenoiserInput | MxArray): MxArray {
    if (inputOrTensor instanceof MxArray || !("hiddenStates" in inputOrTensor)) {
      throw new Error(
        "StableDiffusion3Transformer2DModel.forward: expected a StableDiffusion3DenoiserInput object.",
      );
    }
    const input = inputOrTensor;
    const imageShape = assertImage4d(
      input.hiddenStates,
      "StableDiffusion3Transformer2DModel.forward hiddenStates",
    );
    if (imageShape.channels !== this.#config.inChannels) {
      throw new Error(
        `StableDiffusion3Transformer2DModel.forward: hiddenStates channels must be ${this.#config.inChannels}.`,
      );
    }
    if (
      imageShape.height % this.#config.patchSize !== 0 ||
      imageShape.width % this.#config.patchSize !== 0
    ) {
      throw new Error(
        "StableDiffusion3Transformer2DModel.forward: latent height and width must be divisible by patchSize.",
      );
    }
    const textShape = assertSequence3d(
      input.encoderHiddenStates,
      "StableDiffusion3Transformer2DModel.forward encoderHiddenStates",
    );
    if (
      textShape.batch !== imageShape.batch ||
      textShape.channels !== this.#config.jointAttentionDim
    ) {
      throw new Error(
        `StableDiffusion3Transformer2DModel.forward: encoderHiddenStates must have shape [${imageShape.batch}, length, ${this.#config.jointAttentionDim}], got ${formatShape(
          input.encoderHiddenStates.shape,
        )}.`,
      );
    }
    if (
      input.pooledProjections.shape.length !== 2 ||
      input.pooledProjections.shape[0] !== imageShape.batch ||
      input.pooledProjections.shape[1] !== this.#config.pooledProjectionDim
    ) {
      throw new Error(
        `StableDiffusion3Transformer2DModel.forward: pooledProjections must have shape [${imageShape.batch}, ${this.#config.pooledProjectionDim}], got ${formatShape(
          input.pooledProjections.shape,
        )}.`,
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== imageShape.batch) {
      throw new Error(
        `StableDiffusion3Transformer2DModel.forward: timestep must have shape [${imageShape.batch}], got ${formatShape(
          input.timestep.shape,
        )}.`,
      );
    }

    using imageStates = this.posEmbed.forward(input.hiddenStates);
    using contextStates = this.contextEmbedder.forward(input.encoderHiddenStates);
    using vector = this.timeTextEmbed.forward(input.timestep, input.pooledProjections);
    let image = retainArray(imageStates);
    let context: MxArray | null = retainArray(contextStates);
    try {
      for (let index = 0; index < this.transformerBlocks.length; index += 1) {
        if (context === null) {
          throw new Error(
            "StableDiffusion3Transformer2DModel.forward: context ended before final block.",
          );
        }
        const block = checkedModule(
          this.transformerBlocks,
          index,
          "StableDiffusion3Transformer2DModel.forward transformerBlocks",
        );
        const next = block.run(image, context, vector);
        image.free();
        context.free();
        image = next.hiddenStates;
        context = next.encoderHiddenStates;
      }
      using normalized = this.normOut.forward(image, vector);
      using projected = this.projOut.forward(normalized);
      return unpatchifyStableDiffusion3Latents(
        projected,
        imageShape.height,
        imageShape.width,
        this.#config.patchSize,
        this.#config.outChannels,
      );
    } finally {
      image.free();
      context?.free();
    }
  }

  get config(): StableDiffusion3TransformerConfig {
    return this.#config;
  }
}
