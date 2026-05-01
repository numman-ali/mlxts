import {
  add,
  asType,
  expandDims,
  fastLayerNorm,
  formatShape,
  MxArray,
  multiply,
  random,
  reshape,
  retainArray,
  split,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { LtxVideoTransformerBlock } from "./blocks";
import {
  disposeLtxVideoAdaLayerNormSingleOutput,
  LtxVideoAdaLayerNormSingle,
  LtxVideoCaptionProjection,
} from "./conditioning";
import type { LtxVideoTransformerConfig } from "./config";
import { createLtxVideoRotaryEmbeddings, type LtxRotaryEmbeddings } from "./embeddings";
import type { LtxVideoDenoiserInput } from "./pipeline";
import { applyScaleShift, assertSequence3d, checkedModule, freeArrays } from "./tensor-utils";

type LtxVideoFinalModulation = {
  shift: MxArray;
  scale: MxArray;
};

function scaledNormal(shape: number[], scale: number): MxArray {
  using values = random.normal(shape);
  return multiply(values, scale);
}

function validateLtxVideoTransformerConfig(config: LtxVideoTransformerConfig): void {
  if (config.hiddenSize !== config.numAttentionHeads * config.attentionHeadDim) {
    throw new Error(
      "LtxVideoTransformer3DModel: hiddenSize must equal numAttentionHeads * attentionHeadDim.",
    );
  }
  if (config.patchSize !== 1 || config.patchSizeT !== 1) {
    throw new Error(
      "LtxVideoTransformer3DModel: transformer patch sizes other than 1 are unsupported.",
    );
  }
  if (config.activationFn !== "gelu-approximate") {
    throw new Error("LtxVideoTransformer3DModel: only gelu-approximate activation is supported.");
  }
  if (config.qkNorm !== "rms_norm_across_heads") {
    throw new Error("LtxVideoTransformer3DModel: only rms_norm_across_heads qk_norm is supported.");
  }
  if (config.normElementwiseAffine) {
    throw new Error("LtxVideoTransformer3DModel: affine RMSNorm block weights are unsupported.");
  }
  if (config.crossAttentionDim !== config.hiddenSize) {
    throw new Error("LtxVideoTransformer3DModel: crossAttentionDim must match hiddenSize.");
  }
}

function disposeFinalModulation(modulation: LtxVideoFinalModulation): void {
  modulation.shift.free();
  modulation.scale.free();
}

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function encoderAttentionMask(mask: MxArray, batch: number, textLength: number): MxArray {
  const [maskBatch, maskLength] = mask.shape;
  if (mask.shape.length !== 2 || maskBatch !== batch || maskLength !== textLength) {
    throw new Error(
      `LtxVideoTransformer3DModel.forward: encoderAttentionMask must have shape [${batch}, ${textLength}], got ${formatShape(
        mask.shape,
      )}.`,
    );
  }
  using boolMask = mask.dtype === "bool" ? retainArray(mask) : asType(mask, "bool");
  using keyMask = expandDims(boolMask, 1);
  return expandDims(keyMask, 1);
}

function rotaryCacheKey(input: LtxVideoDenoiserInput, batch: number): string {
  return [
    batch,
    input.numFrames,
    input.height,
    input.width,
    input.ropeInterpolationScale[0],
    input.ropeInterpolationScale[1],
    input.ropeInterpolationScale[2],
  ].join(":");
}

/** Diffusers-compatible classic LTX-Video `LTXVideoTransformer3DModel` tensor path. */
export class LtxVideoTransformer3DModel extends Module {
  projIn: Linear;
  scaleShiftTable: MxArray;
  timeEmbed: LtxVideoAdaLayerNormSingle;
  captionProjection: LtxVideoCaptionProjection;
  transformerBlocks: LtxVideoTransformerBlock[];
  normOut: null;
  projOut: Linear;
  #config: LtxVideoTransformerConfig;
  #rotaryCache = new Map<string, LtxRotaryEmbeddings>();

  constructor(config: LtxVideoTransformerConfig) {
    super();
    validateLtxVideoTransformerConfig(config);
    this.#config = config;
    this.projIn = new Linear(config.inChannels, config.hiddenSize);
    this.scaleShiftTable = scaledNormal([2, config.hiddenSize], config.hiddenSize ** -0.5);
    this.timeEmbed = new LtxVideoAdaLayerNormSingle(config.hiddenSize);
    this.captionProjection = new LtxVideoCaptionProjection(
      config.captionChannels,
      config.hiddenSize,
    );
    this.transformerBlocks = Array.from(
      { length: config.numLayers },
      () => new LtxVideoTransformerBlock(config),
    );
    this.normOut = null;
    this.projOut = new Linear(config.hiddenSize, config.outChannels);
  }

  forward(input: LtxVideoDenoiserInput): MxArray;
  forward(...args: MxArray[]): MxArray;
  /** Run an LTX-Video denoising prediction over packed video latents. */
  forward(inputOrTensor: LtxVideoDenoiserInput | MxArray): MxArray {
    if (inputOrTensor instanceof MxArray || !("hiddenStates" in inputOrTensor)) {
      throw new Error(
        "LtxVideoTransformer3DModel.forward: expected an LtxVideoDenoiserInput object.",
      );
    }
    const input = inputOrTensor;
    const hiddenShape = assertSequence3d(
      input.hiddenStates,
      "LtxVideoTransformer3DModel.forward hiddenStates",
    );
    const textShape = assertSequence3d(
      input.encoderHiddenStates,
      "LtxVideoTransformer3DModel.forward encoderHiddenStates",
    );
    this.#validateInput(input, hiddenShape, textShape);

    using projectedHidden = this.projIn.forward(input.hiddenStates);
    const time = this.timeEmbed.embed(input.timestep, projectedHidden.dtype);
    const mask =
      input.encoderAttentionMask === undefined
        ? null
        : encoderAttentionMask(input.encoderAttentionMask, hiddenShape.batch, textShape.length);
    try {
      using projectedEncoder = this.captionProjection.forward(input.encoderHiddenStates);
      const rotaryEmbeddings = this.#rotaryEmbeddings(input, hiddenShape.batch);
      let hidden = retainArray(projectedHidden);
      try {
        for (let index = 0; index < this.transformerBlocks.length; index += 1) {
          const block = checkedModule(
            this.transformerBlocks,
            index,
            "LtxVideoTransformer3DModel.forward transformerBlocks",
          );
          const next = block.run(
            hidden,
            projectedEncoder,
            time.modulation,
            rotaryEmbeddings,
            mask ?? undefined,
          );
          hidden.free();
          hidden = next;
        }
        const finalModulation = this.#finalModulation(time.embeddedTimestep, hiddenShape.batch);
        try {
          using normalized = fastLayerNorm(hidden, undefined, undefined, { eps: 1e-6 });
          using modulated = applyScaleShift(
            normalized,
            finalModulation.shift,
            finalModulation.scale,
          );
          return this.projOut.forward(modulated);
        } finally {
          disposeFinalModulation(finalModulation);
        }
      } finally {
        hidden.free();
      }
    } finally {
      mask?.free();
      disposeLtxVideoAdaLayerNormSingleOutput(time);
    }
  }

  get config(): LtxVideoTransformerConfig {
    return this.#config;
  }

  override [Symbol.dispose](): void {
    for (const embeddings of this.#rotaryCache.values()) {
      embeddings.cos.free();
      embeddings.sin.free();
    }
    this.#rotaryCache.clear();
    super[Symbol.dispose]();
  }

  #validateInput(
    input: LtxVideoDenoiserInput,
    hiddenShape: { batch: number; length: number; channels: number },
    textShape: { batch: number; length: number; channels: number },
  ): void {
    if (hiddenShape.channels !== this.#config.inChannels) {
      throw new Error(
        `LtxVideoTransformer3DModel.forward: hiddenStates channels must be ${this.#config.inChannels}.`,
      );
    }
    if (hiddenShape.length !== input.numFrames * input.height * input.width) {
      throw new Error(
        "LtxVideoTransformer3DModel.forward: numFrames * height * width must match hiddenStates length.",
      );
    }
    if (
      textShape.batch !== hiddenShape.batch ||
      textShape.channels !== this.#config.captionChannels
    ) {
      throw new Error(
        `LtxVideoTransformer3DModel.forward: encoderHiddenStates must have shape [${hiddenShape.batch}, length, ${this.#config.captionChannels}], got ${formatShape(
          input.encoderHiddenStates.shape,
        )}.`,
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== hiddenShape.batch) {
      throw new Error(
        `LtxVideoTransformer3DModel.forward: timestep must have shape [${hiddenShape.batch}], got ${formatShape(
          input.timestep.shape,
        )}.`,
      );
    }
  }

  #rotaryEmbeddings(input: LtxVideoDenoiserInput, batch: number): LtxRotaryEmbeddings {
    const key = rotaryCacheKey(input, batch);
    const cached = this.#rotaryCache.get(key);
    if (cached !== undefined) {
      return cached;
    }
    const embeddings = createLtxVideoRotaryEmbeddings({
      batchSize: batch,
      latentFrames: input.numFrames,
      latentHeight: input.height,
      latentWidth: input.width,
      dim: this.#config.hiddenSize,
      patchSize: this.#config.patchSize,
      patchSizeT: this.#config.patchSizeT,
      ropeInterpolationScale: input.ropeInterpolationScale,
    });
    this.#rotaryCache.set(key, embeddings);
    return embeddings;
  }

  #finalModulation(embeddedTimestep: MxArray, batch: number): LtxVideoFinalModulation {
    if (
      embeddedTimestep.shape.length !== 3 ||
      embeddedTimestep.shape[0] !== batch ||
      embeddedTimestep.shape[1] !== 1 ||
      embeddedTimestep.shape[2] !== this.#config.hiddenSize
    ) {
      throw new Error("LtxVideoTransformer3DModel.finalModulation: timestep shape mismatch.");
    }
    using table = reshape(this.scaleShiftTable, [1, 2, this.#config.hiddenSize]);
    using values = add(table, embeddedTimestep);
    const parts = split(values, 2, 1);
    try {
      return {
        shift: retainArray(partAt(parts, 0, "LtxVideoTransformer3DModel.finalModulation")),
        scale: retainArray(partAt(parts, 1, "LtxVideoTransformer3DModel.finalModulation")),
      };
    } finally {
      freeArrays(parts);
    }
  }
}
