import type { MxArray } from "@mlxts/core";
import { asType, concatenate, formatShape, multiply, ones, retainArray, zeros } from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import { QwenImageAdaptiveLayerNormContinuous, QwenImageTransformerBlock } from "./blocks";
import type { QwenImageTransformerConfig } from "./config";
import { QwenImageRopeEmbedder, QwenImageTimestepProjEmbeddings } from "./embeddings";
import {
  type QwenImageRopeImageShape,
  type QwenImageRopeImageShapes,
  qwenImageRopeImageShapeProduct,
  qwenImageRopeImageShapesProduct,
} from "./latents";
import { assertSequence3d, checkedModule, sliceAxis } from "./tensor-utils";

export type QwenImageDenoiserInput = {
  hiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  timestep: MxArray;
  imageShape: QwenImageRopeImageShape;
  imageShapes?: QwenImageRopeImageShapes;
  encoderHiddenStatesMask?: MxArray;
  additionalTCond?: MxArray;
};

function validateQwenImageTransformerConfig(config: QwenImageTransformerConfig): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "QwenImageTransformer2DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  if (config.guidanceEmbeds) {
    throw new Error("QwenImageTransformer2DModel: guidance_embeds is unsupported.");
  }
  if (config.useAdditionalTCond) {
    throw new Error("QwenImageTransformer2DModel: use_additional_t_cond is unsupported.");
  }
  if (config.useLayer3dRope) {
    throw new Error("QwenImageTransformer2DModel: use_layer3d_rope is unsupported.");
  }
}

function sameImageShape(left: QwenImageRopeImageShape, right: QwenImageRopeImageShape): boolean {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function resolveImageShapes(input: QwenImageDenoiserInput): QwenImageRopeImageShapes {
  const imageShapes = input.imageShapes ?? [input.imageShape];
  if (imageShapes.length === 0) {
    throw new Error("QwenImageTransformer2DModel.forward: imageShapes must not be empty.");
  }
  const firstShape = imageShapes[0];
  if (firstShape === undefined || !sameImageShape(input.imageShape, firstShape)) {
    throw new Error(
      "QwenImageTransformer2DModel.forward: first imageShapes entry must match imageShape.",
    );
  }
  return imageShapes;
}

function zeroCondTimestep(timestep: MxArray): MxArray {
  using zeroTimestep = multiply(timestep, 0);
  return concatenate([timestep, zeroTimestep], 0);
}

function createModulateIndex(
  imageShapes: QwenImageRopeImageShapes,
  batchSize: number,
  dtype: MxArray["dtype"],
): MxArray {
  const targetShape = imageShapes[0];
  if (targetShape === undefined) {
    throw new Error("QwenImageTransformer2DModel.forward: imageShapes must not be empty.");
  }
  const targetLength = qwenImageRopeImageShapeProduct(targetShape);
  const referenceLength = qwenImageRopeImageShapesProduct(imageShapes.slice(1));
  using targetIndex = zeros([batchSize, targetLength], dtype);
  if (referenceLength === 0) {
    return retainArray(targetIndex);
  }
  using referenceIndex = ones([batchSize, referenceLength], dtype);
  return concatenate([targetIndex, referenceIndex], 1);
}

/** Diffusers-compatible base Qwen-Image `QwenImageTransformer2DModel` tensor path. */
export class QwenImageTransformer2DModel extends Module {
  posEmbed: QwenImageRopeEmbedder;
  timeTextEmbed: QwenImageTimestepProjEmbeddings;
  txtNorm: RMSNorm;
  imgIn: Linear;
  txtIn: Linear;
  transformerBlocks: QwenImageTransformerBlock[];
  normOut: QwenImageAdaptiveLayerNormContinuous;
  projOut: Linear;
  #config: QwenImageTransformerConfig;

  constructor(config: QwenImageTransformerConfig) {
    super();
    validateQwenImageTransformerConfig(config);
    this.#config = config;
    this.posEmbed = new QwenImageRopeEmbedder(
      config.attentionHeadDim,
      config.ropeTheta,
      config.axesDimsRope,
    );
    this.timeTextEmbed = new QwenImageTimestepProjEmbeddings(
      config.hiddenSize,
      config.useAdditionalTCond,
    );
    this.txtNorm = new RMSNorm(config.jointAttentionDim, 1e-6);
    this.imgIn = new Linear(config.inChannels, config.hiddenSize);
    this.txtIn = new Linear(config.jointAttentionDim, config.hiddenSize);
    this.transformerBlocks = Array.from(
      { length: config.numLayers },
      () =>
        new QwenImageTransformerBlock({
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          headDim: config.attentionHeadDim,
        }),
    );
    this.normOut = new QwenImageAdaptiveLayerNormContinuous(config.hiddenSize);
    this.projOut = new Linear(config.hiddenSize, config.packedLatentChannels);
  }

  forward(input: QwenImageDenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run a Qwen-Image denoising prediction over packed latents and Qwen text embeddings. */
  forward(inputOrTensor: QwenImageDenoiserInput | MxArray): MxArray {
    if (!("hiddenStates" in inputOrTensor)) {
      throw new Error(
        "QwenImageTransformer2DModel.forward: expected a QwenImageDenoiserInput object.",
      );
    }
    const input = inputOrTensor;
    const imageShape = assertSequence3d(
      input.hiddenStates,
      "QwenImageTransformer2DModel.forward hiddenStates",
    );
    const textShape = assertSequence3d(
      input.encoderHiddenStates,
      "QwenImageTransformer2DModel.forward encoderHiddenStates",
    );
    if (imageShape.channels !== this.#config.inChannels) {
      throw new Error(
        `QwenImageTransformer2DModel.forward: hiddenStates channels must be ${
          this.#config.inChannels
        }.`,
      );
    }
    if (
      textShape.batch !== imageShape.batch ||
      textShape.channels !== this.#config.jointAttentionDim
    ) {
      throw new Error(
        `QwenImageTransformer2DModel.forward: encoderHiddenStates must have shape [${
          imageShape.batch
        }, length, ${this.#config.jointAttentionDim}], got ${formatShape(
          input.encoderHiddenStates.shape,
        )}.`,
      );
    }
    const imageShapes = resolveImageShapes(input);
    if (qwenImageRopeImageShapesProduct(imageShapes) !== imageShape.length) {
      throw new Error(
        "QwenImageTransformer2DModel.forward: imageShapes product must match image sequence " +
          "length.",
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== imageShape.batch) {
      throw new Error(
        `QwenImageTransformer2DModel.forward: timestep must have shape [${
          imageShape.batch
        }], got ${formatShape(input.timestep.shape)}.`,
      );
    }

    using imageStates = this.imgIn.forward(input.hiddenStates);
    using normalizedText = this.txtNorm.forward(input.encoderHiddenStates);
    using textStates = this.txtIn.forward(normalizedText);
    using timestep = asType(input.timestep, input.hiddenStates.dtype);
    using embeddingTimestep = this.#config.zeroCondT
      ? zeroCondTimestep(timestep)
      : retainArray(timestep);
    using vector = this.timeTextEmbed.embed(
      embeddingTimestep,
      input.hiddenStates.dtype,
      input.additionalTCond,
    );
    using textVector = this.#config.zeroCondT
      ? sliceAxis(vector, 0, 0, imageShape.batch)
      : retainArray(vector);
    using rope = this.posEmbed.embed(imageShapes, textShape.length, input.hiddenStates.dtype);
    const modulateIndex = this.#config.zeroCondT
      ? createModulateIndex(imageShapes, imageShape.batch, "int32")
      : null;

    let image = retainArray(imageStates);
    let text = retainArray(textStates);
    try {
      for (let index = 0; index < this.transformerBlocks.length; index += 1) {
        const block = checkedModule(
          this.transformerBlocks,
          index,
          "QwenImageTransformer2DModel.forward transformerBlocks",
        );
        const next = block.run(
          image,
          text,
          vector,
          textVector,
          rope,
          input.encoderHiddenStatesMask,
          modulateIndex ?? undefined,
        );
        image.free();
        text.free();
        image = next.image;
        text = next.text;
      }
      using normalizedOutput = this.normOut.forward(image, textVector);
      return this.projOut.forward(normalizedOutput);
    } finally {
      modulateIndex?.free();
      image.free();
      text.free();
    }
  }

  get config(): QwenImageTransformerConfig {
    return this.#config;
  }
}
