/**
 * Stable Diffusion UNet2DConditionModel block modules.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, concatenate, formatShape, reshape, retainArray } from "@mlxts/core";
import { Conv2d, GroupNorm, Linear, Module, silu } from "@mlxts/nn";

import { upsampleNearest2d } from "./spatial";
import { StableDiffusionUNetTransformer2d } from "./unet-transformer";

export type StableDiffusionUNetBlockForwardResult = {
  hidden: MxArray;
  residuals: MxArray[];
};

function assertImage4d(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, height, width, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-4 NHWC input, got ${formatShape(x.shape)}.`);
  }
  return [batch, height, width, channels];
}

function checkedModule<T>(modules: readonly T[], index: number, owner: string): T {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

function freeAll(values: readonly MxArray[]): void {
  for (const value of values) {
    value.free();
  }
}

function popResidual(residuals: MxArray[], owner: string): MxArray {
  const residual = residuals.pop();
  if (residual === undefined) {
    throw new Error(`${owner}: missing residual hidden state.`);
  }
  return residual;
}

/** Residual block used by Stable Diffusion UNet down, mid, and up blocks. */
export class StableDiffusionUNetResnetBlock2d extends Module {
  norm1: GroupNorm;
  conv1: Conv2d;
  timeEmbeddingProjection: Linear;
  norm2: GroupNorm;
  conv2: Conv2d;
  convShortcut: Conv2d | null;
  #outChannels: number;

  constructor(
    inChannels: number,
    outChannels: number,
    timeEmbedDims: number,
    normGroups: number,
    normEps: number,
  ) {
    super();
    this.norm1 = new GroupNorm(normGroups, inChannels, normEps);
    this.conv1 = new Conv2d(inChannels, outChannels, 3, 1, 1);
    this.timeEmbeddingProjection = new Linear(timeEmbedDims, outChannels);
    this.norm2 = new GroupNorm(normGroups, outChannels, normEps);
    this.conv2 = new Conv2d(outChannels, outChannels, 3, 1, 1);
    this.convShortcut =
      inChannels === outChannels ? null : new Conv2d(inChannels, outChannels, 1, 1, 0);
    this.#outChannels = outChannels;
  }

  /** Run the residual block over an NHWC feature map with batch-aligned time embeddings. */
  forward(x: MxArray, timeEmbedding: MxArray): MxArray {
    const [batch] = assertImage4d(x, "StableDiffusionUNetResnetBlock2d.forward");
    using normalized = this.norm1.forward(x);
    using activated = silu(normalized);
    using first = this.conv1.forward(activated);
    using activatedTime = silu(timeEmbedding);
    using projectedTime = this.timeEmbeddingProjection.forward(activatedTime);
    using timeView = reshape(projectedTime, [batch, 1, 1, this.#outChannels]);
    using conditioned = add(first, timeView);
    using normalizedConditioned = this.norm2.forward(conditioned);
    using activatedConditioned = silu(normalizedConditioned);
    using residual = this.conv2.forward(activatedConditioned);

    if (this.convShortcut === null) {
      return add(x, residual);
    }

    using shortcut = this.convShortcut.forward(x);
    return add(shortcut, residual);
  }
}

/** Down block used by Stable Diffusion UNet2DConditionModel. */
export class StableDiffusionUNetDownBlock2d extends Module {
  resnets: StableDiffusionUNetResnetBlock2d[];
  attentions: StableDiffusionUNetTransformer2d[] | null;
  downsample: Conv2d | null;

  constructor(options: {
    inChannels: number;
    outChannels: number;
    timeEmbedDims: number;
    layers: number;
    transformerLayers: number;
    numHeads: number;
    crossAttentionDims: number;
    normGroups: number;
    normEps: number;
    addDownsample: boolean;
    addCrossAttention: boolean;
    useLinearProjection: boolean;
  }) {
    super();
    this.resnets = Array.from({ length: options.layers }, (_, index) => {
      const inputChannels = index === 0 ? options.inChannels : options.outChannels;
      return new StableDiffusionUNetResnetBlock2d(
        inputChannels,
        options.outChannels,
        options.timeEmbedDims,
        options.normGroups,
        options.normEps,
      );
    });
    this.attentions = options.addCrossAttention
      ? Array.from(
          { length: options.layers },
          () =>
            new StableDiffusionUNetTransformer2d({
              channels: options.outChannels,
              crossAttentionDims: options.crossAttentionDims,
              numHeads: options.numHeads,
              layers: options.transformerLayers,
              normGroups: options.normGroups,
              useLinearProjection: options.useLinearProjection,
            }),
        )
      : null;
    this.downsample = options.addDownsample
      ? new Conv2d(options.outChannels, options.outChannels, 3, 2, 1)
      : null;
  }

  forward(..._args: MxArray[]): MxArray {
    throw new Error("StableDiffusionUNetDownBlock2d.forward: use run() with conditioning tensors.");
  }

  /** Run the down block and return retained skip residuals for the later up path. */
  run(
    x: MxArray,
    timeEmbedding: MxArray,
    encoderHiddenStates: MxArray,
  ): StableDiffusionUNetBlockForwardResult {
    const residuals: MxArray[] = [];
    let hidden = x;
    let ownsHidden = false;
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const resnet = checkedModule(this.resnets, index, "StableDiffusionUNetDownBlock2d.forward");
        const resnetOutput = resnet.forward(hidden, timeEmbedding);
        if (ownsHidden) {
          hidden.free();
        }
        hidden = resnetOutput;
        ownsHidden = true;

        if (this.attentions !== null) {
          const attention = checkedModule(
            this.attentions,
            index,
            "StableDiffusionUNetDownBlock2d.forward",
          );
          const attended = attention.forward(hidden, encoderHiddenStates);
          hidden.free();
          hidden = attended;
        }
        residuals.push(retainArray(hidden));
      }

      if (this.downsample !== null) {
        const downsampled = this.downsample.forward(hidden);
        if (ownsHidden) {
          hidden.free();
        }
        hidden = downsampled;
        ownsHidden = true;
        residuals.push(retainArray(hidden));
      }

      return { hidden, residuals };
    } catch (error) {
      if (ownsHidden) {
        hidden.free();
      }
      freeAll(residuals);
      throw error;
    }
  }
}

/** Up block used by Stable Diffusion UNet2DConditionModel. */
export class StableDiffusionUNetUpBlock2d extends Module {
  resnets: StableDiffusionUNetResnetBlock2d[];
  attentions: StableDiffusionUNetTransformer2d[] | null;
  upsample: Conv2d | null;

  constructor(options: {
    inChannels: number;
    outChannels: number;
    previousOutputChannels: number;
    timeEmbedDims: number;
    layers: number;
    transformerLayers: number;
    numHeads: number;
    crossAttentionDims: number;
    normGroups: number;
    normEps: number;
    addUpsample: boolean;
    addCrossAttention: boolean;
    useLinearProjection: boolean;
  }) {
    super();
    this.resnets = Array.from({ length: options.layers }, (_, index) => {
      const skipChannels = index === options.layers - 1 ? options.inChannels : options.outChannels;
      const hiddenChannels = index === 0 ? options.previousOutputChannels : options.outChannels;
      return new StableDiffusionUNetResnetBlock2d(
        hiddenChannels + skipChannels,
        options.outChannels,
        options.timeEmbedDims,
        options.normGroups,
        options.normEps,
      );
    });
    this.attentions = options.addCrossAttention
      ? Array.from(
          { length: options.layers },
          () =>
            new StableDiffusionUNetTransformer2d({
              channels: options.outChannels,
              crossAttentionDims: options.crossAttentionDims,
              numHeads: options.numHeads,
              layers: options.transformerLayers,
              normGroups: options.normGroups,
              useLinearProjection: options.useLinearProjection,
            }),
        )
      : null;
    this.upsample = options.addUpsample
      ? new Conv2d(options.outChannels, options.outChannels, 3, 1, 1)
      : null;
  }

  forward(..._args: MxArray[]): MxArray {
    throw new Error("StableDiffusionUNetUpBlock2d.forward: use run() with residual state.");
  }

  /** Run the up block, consuming residuals from the end of the shared residual stack. */
  run(
    x: MxArray,
    residuals: MxArray[],
    timeEmbedding: MxArray,
    encoderHiddenStates: MxArray,
  ): MxArray {
    let hidden = x;
    let ownsHidden = false;
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        using residual = popResidual(residuals, "StableDiffusionUNetUpBlock2d.forward");
        using concatenated = concatenate([hidden, residual], -1);
        const resnet = checkedModule(this.resnets, index, "StableDiffusionUNetUpBlock2d.forward");
        const resnetOutput = resnet.forward(concatenated, timeEmbedding);
        if (ownsHidden) {
          hidden.free();
        }
        hidden = resnetOutput;
        ownsHidden = true;

        if (this.attentions !== null) {
          const attention = checkedModule(
            this.attentions,
            index,
            "StableDiffusionUNetUpBlock2d.forward",
          );
          const attended = attention.forward(hidden, encoderHiddenStates);
          hidden.free();
          hidden = attended;
        }
      }

      if (this.upsample === null) {
        return hidden;
      }

      using upsampled = upsampleNearest2d(hidden);
      const output = this.upsample.forward(upsampled);
      if (ownsHidden) {
        hidden.free();
      }
      return output;
    } catch (error) {
      if (ownsHidden) {
        hidden.free();
      }
      throw error;
    }
  }
}

/** Mid block used by Stable Diffusion UNet2DConditionModel. */
export class StableDiffusionUNetMidBlock2d extends Module {
  resnetIn: StableDiffusionUNetResnetBlock2d;
  attention: StableDiffusionUNetTransformer2d;
  resnetOut: StableDiffusionUNetResnetBlock2d;

  constructor(
    channels: number,
    timeEmbedDims: number,
    transformerLayers: number,
    numHeads: number,
    crossAttentionDims: number,
    normGroups: number,
    normEps: number,
    useLinearProjection: boolean,
  ) {
    super();
    this.resnetIn = new StableDiffusionUNetResnetBlock2d(
      channels,
      channels,
      timeEmbedDims,
      normGroups,
      normEps,
    );
    this.attention = new StableDiffusionUNetTransformer2d({
      channels,
      crossAttentionDims,
      numHeads,
      layers: transformerLayers,
      normGroups,
      useLinearProjection,
    });
    this.resnetOut = new StableDiffusionUNetResnetBlock2d(
      channels,
      channels,
      timeEmbedDims,
      normGroups,
      normEps,
    );
  }

  /** Run the Stable Diffusion cross-attention UNet mid block. */
  forward(x: MxArray, timeEmbedding: MxArray, encoderHiddenStates: MxArray): MxArray {
    using first = this.resnetIn.forward(x, timeEmbedding);
    using attended = this.attention.forward(first, encoderHiddenStates);
    return this.resnetOut.forward(attended, timeEmbedding);
  }
}
