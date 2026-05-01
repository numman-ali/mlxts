/**
 * Z-Image transformer denoiser.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  concatenate,
  expandDims,
  formatShape,
  multiply,
  retainArray,
  squeeze,
  where,
  zeros,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { ZImageFinalLayer, ZImageTransformerBlock } from "./blocks";
import type { ZImagePatchGeometry, ZImageTransformerConfig } from "./config";
import { ZImageCaptionEmbedder, ZImageRopeEmbedder, ZImageTimestepEmbedder } from "./embeddings";
import {
  padZImageFeature,
  patchifyZImageLatent,
  unpatchifyZImageLatent,
  type ZImagePaddedFeature,
} from "./latents";
import { checkedModule, sliceAxis } from "./tensor-utils";

const Z_IMAGE_ADALN_EMBED_DIM = 256;

export type ZImageDenoiserInput = {
  latents: readonly MxArray[];
  captionFeatures: readonly MxArray[];
  timestep: MxArray;
  patchSize?: number;
  framePatchSize?: number;
};

type PreparedSequence = {
  hidden: MxArray;
  rope: MxArray;
  length: number;
};

function disposePaddedFeature(feature: ZImagePaddedFeature): void {
  feature.features.free();
  feature.positionIds.free();
  feature.padMask.free();
}

function disposePreparedSequence(sequence: PreparedSequence): void {
  sequence.hidden.free();
  sequence.rope.free();
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function paddedLength(length: number, multiple: number): number {
  return length + ((multiple - (length % multiple)) % multiple);
}

function validateZImageTransformerConfig(config: ZImageTransformerConfig): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "ZImageTransformer2DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  const ropeDims = config.axesDims.reduce((sum, dim) => sum + dim, 0);
  if (ropeDims !== config.attentionHeadDim) {
    throw new Error("ZImageTransformer2DModel: axesDims sum must equal attentionHeadDim.");
  }
  if (config.siglipFeatureDim !== null) {
    throw new Error(
      "ZImageTransformer2DModel: SigLIP/Omni configs are not supported in this runtime.",
    );
  }
}

function geometryKey(geometry: Pick<ZImagePatchGeometry, "patchSize" | "framePatchSize">): string {
  return `${geometry.patchSize}-${geometry.framePatchSize}`;
}

/** Diffusers-compatible base Z-Image `ZImageTransformer2DModel` tensor path. */
export class ZImageTransformer2DModel extends Module {
  imageEmbedders: Linear[];
  finalLayers: ZImageFinalLayer[];
  noiseRefiner: ZImageTransformerBlock[];
  contextRefiner: ZImageTransformerBlock[];
  timeEmbedding: ZImageTimestepEmbedder;
  captionEmbedder: ZImageCaptionEmbedder;
  xPadToken: MxArray;
  capPadToken: MxArray;
  layers: ZImageTransformerBlock[];
  positionalEmbedding: ZImageRopeEmbedder;
  #config: ZImageTransformerConfig;
  #adalnDims: number;

  constructor(config: ZImageTransformerConfig) {
    super();
    validateZImageTransformerConfig(config);
    this.#config = config;
    this.#adalnDims = Math.min(config.hiddenSize, Z_IMAGE_ADALN_EMBED_DIM);
    this.imageEmbedders = config.patchGeometries.map(
      (geometry) => new Linear(geometry.packedLatentChannels, config.hiddenSize),
    );
    this.finalLayers = config.patchGeometries.map(
      (geometry) =>
        new ZImageFinalLayer(config.hiddenSize, this.#adalnDims, geometry.packedLatentChannels),
    );
    this.noiseRefiner = Array.from({ length: config.numRefinerLayers }, () =>
      this.#createBlock(true),
    );
    this.contextRefiner = Array.from({ length: config.numRefinerLayers }, () =>
      this.#createBlock(false),
    );
    this.timeEmbedding = new ZImageTimestepEmbedder(this.#adalnDims);
    this.captionEmbedder = new ZImageCaptionEmbedder(
      config.captionFeatureDim,
      config.hiddenSize,
      config.normEps,
    );
    this.xPadToken = zeros([1, config.hiddenSize]);
    this.capPadToken = zeros([1, config.hiddenSize]);
    this.layers = Array.from({ length: config.numLayers }, () => this.#createBlock(true));
    this.positionalEmbedding = new ZImageRopeEmbedder(
      config.attentionHeadDim,
      config.ropeTheta,
      config.axesDims,
      config.axesLens,
    );
  }

  forward(input: ZImageDenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run a base Z-Image denoising prediction over latent samples and caption features. */
  forward(inputOrTensor: ZImageDenoiserInput | MxArray): MxArray {
    if (!("latents" in inputOrTensor)) {
      throw new Error("ZImageTransformer2DModel.forward: expected a ZImageDenoiserInput object.");
    }
    const input = inputOrTensor;
    if (input.latents.length !== 1 || input.captionFeatures.length !== 1) {
      throw new Error("ZImageTransformer2DModel.forward: this runtime supports batch size 1.");
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== 1) {
      throw new Error(
        `ZImageTransformer2DModel.forward: timestep must have shape [1], got ${formatShape(input.timestep.shape)}.`,
      );
    }
    const geometryIndex = this.#geometryIndex(input.patchSize ?? 2, input.framePatchSize ?? 1);
    const geometry = checkedModule(
      this.#config.patchGeometries,
      geometryIndex,
      "ZImageTransformer2DModel.forward geometry",
    );
    const imageEmbedder = checkedModule(
      this.imageEmbedders,
      geometryIndex,
      "ZImageTransformer2DModel.forward imageEmbedders",
    );
    const finalLayer = checkedModule(
      this.finalLayers,
      geometryIndex,
      "ZImageTransformer2DModel.forward finalLayers",
    );
    const latent = checkedModule(input.latents, 0, "ZImageTransformer2DModel.forward latents");
    const captionFeature = checkedModule(
      input.captionFeatures,
      0,
      "ZImageTransformer2DModel.forward captionFeatures",
    );

    using scaledTimestep = multiply(input.timestep, this.#config.timestepScale);
    using adalnInput = this.timeEmbedding.embed(scaledTimestep, latent.dtype);
    const patchified = patchifyZImageLatent(latent, geometry);
    let imagePadded: ZImagePaddedFeature | null = null;
    let captionPadded: ZImagePaddedFeature | null = null;
    let imagePrepared: PreparedSequence | null = null;
    let captionPrepared: PreparedSequence | null = null;
    try {
      const captionLength = captionFeature.shape[0];
      if (captionLength === undefined) {
        throw new Error(
          "ZImageTransformer2DModel.forward: caption features are missing a length dimension.",
        );
      }
      const captionTotalLength = paddedLength(captionLength, this.#config.sequenceMultiple);
      captionPadded = padZImageFeature(
        captionFeature,
        this.#config.sequenceMultiple,
        [captionTotalLength, 1, 1],
        [1, 0, 0],
      );
      imagePadded = padZImageFeature(
        patchified.patches,
        this.#config.sequenceMultiple,
        [patchified.tokenGrid.frames, patchified.tokenGrid.height, patchified.tokenGrid.width],
        [captionPadded.totalLength + 1, 0, 0],
      );

      using imageProjected = imageEmbedder.forward(imagePadded.features);
      imagePrepared = this.#prepareSequence(
        imageProjected,
        imagePadded,
        this.xPadToken,
        latent.dtype,
      );
      const imageHidden = this.#runBlocks(
        this.noiseRefiner,
        imagePrepared.hidden,
        imagePrepared.rope,
        adalnInput,
      );
      try {
        using captionProjected = this.captionEmbedder.forward(captionPadded.features);
        captionPrepared = this.#prepareSequence(
          captionProjected,
          captionPadded,
          this.capPadToken,
          latent.dtype,
        );
        const captionHidden = this.#runBlocks(
          this.contextRefiner,
          captionPrepared.hidden,
          captionPrepared.rope,
        );
        try {
          using unified = concatenate([imageHidden, captionHidden], 1);
          using unifiedRope = concatenate([imagePrepared.rope, captionPrepared.rope], 2);
          using mainHidden = this.#runBlocks(this.layers, unified, unifiedRope, adalnInput);
          using projected = finalLayer.forward(mainHidden, adalnInput);
          using firstBatch = sliceAxis(projected, 0, 0, 1);
          using sequence = squeeze(firstBatch, 0);
          using sample = unpatchifyZImageLatent(
            sequence,
            patchified.size,
            geometry,
            this.#config.outChannels,
          );
          return expandDims(sample, 0);
        } finally {
          captionHidden.free();
        }
      } finally {
        imageHidden.free();
      }
    } finally {
      patchified.patches.free();
      if (imagePrepared !== null) {
        disposePreparedSequence(imagePrepared);
      }
      if (captionPrepared !== null) {
        disposePreparedSequence(captionPrepared);
      }
      if (imagePadded !== null) {
        disposePaddedFeature(imagePadded);
      }
      if (captionPadded !== null) {
        disposePaddedFeature(captionPadded);
      }
    }
  }

  #createBlock(modulation: boolean): ZImageTransformerBlock {
    return new ZImageTransformerBlock({
      hiddenSize: this.#config.hiddenSize,
      numHeads: this.#config.numAttentionHeads,
      normEps: this.#config.normEps,
      qkNorm: this.#config.qkNorm,
      modulation,
      adalnDims: this.#adalnDims,
    });
  }

  #geometryIndex(patchSize: number, framePatchSize: number): number {
    positiveInteger(patchSize, "patchSize");
    positiveInteger(framePatchSize, "framePatchSize");
    const index = this.#config.patchGeometries.findIndex(
      (geometry) => geometry.patchSize === patchSize && geometry.framePatchSize === framePatchSize,
    );
    if (index === -1) {
      throw new Error(
        `ZImageTransformer2DModel.forward: unsupported patch geometry ${geometryKey({
          patchSize,
          framePatchSize,
        })}.`,
      );
    }
    return index;
  }

  #prepareSequence(
    hidden: MxArray,
    padded: ZImagePaddedFeature,
    padToken: MxArray,
    dtype: MxArray["dtype"],
  ): PreparedSequence {
    using padMask = expandDims(padded.padMask, 1);
    using paddedHidden = where(padMask, padToken, hidden);
    using batched = expandDims(paddedHidden, 0);
    using fullRope = this.positionalEmbedding.embed(padded.positionIds, dtype);
    return {
      hidden: retainArray(batched),
      rope: sliceAxis(fullRope, 2, 0, padded.totalLength),
      length: padded.totalLength,
    };
  }

  #runBlocks(
    blocks: readonly ZImageTransformerBlock[],
    initialHidden: MxArray,
    rope: MxArray,
    adalnInput?: MxArray,
  ): MxArray {
    let hidden = retainArray(initialHidden);
    try {
      for (let index = 0; index < blocks.length; index += 1) {
        const block = checkedModule(blocks, index, "ZImageTransformer2DModel.runBlocks");
        const nextHidden = block.forward(hidden, rope, adalnInput);
        hidden.free();
        hidden = nextHidden;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }

  get config(): ZImageTransformerConfig {
    return this.#config;
  }
}
