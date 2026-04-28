/**
 * Qwen 3.5 vision tower for image-conditioned prompts.
 * @module
 */

import {
  add,
  arange,
  array,
  asType,
  concatenate,
  cos,
  expandDims,
  formatShape,
  type MxArray,
  matmul,
  multiply,
  random,
  reshape,
  retainArray,
  scaledDotProductAttention,
  sin,
  squeeze,
  sum,
  takeAxis,
  transpose,
  zeros,
} from "@mlxts/core";
import { Embedding, gelu, LayerNorm, Linear, Module } from "@mlxts/nn";

import type { Qwen3_5VisionConfig } from "./types";
import {
  applyVisionRotaryPosEmb,
  bilinearInterpolationTables,
  flattenPairEmbeddings,
  type GridThw,
  gridGeometry,
  gridThwList,
  reorderedPatchIndices,
  repeatFrames,
  rotaryPairIndices,
  takeAxis1Slice,
  takeSequenceSlice,
} from "./vision-support";

/** Flat-patch embedding projection matching Qwen's 3D patch kernel layout. */
export class Qwen3_5VisionPatchEmbed extends Module {
  weight: MxArray;
  bias: MxArray;
  #flatPatchDim: number;
  #embedDim: number;
  #transposedWeight: MxArray | null = null;
  #transposedWeightSource: MxArray | null = null;

  constructor(config: Qwen3_5VisionConfig) {
    super();
    const patchSize =
      typeof config.patchSize === "number" ? config.patchSize : (config.patchSize[0] ?? 0);
    const temporalPatchSize =
      typeof config.temporalPatchSize === "number"
        ? config.temporalPatchSize
        : (config.temporalPatchSize[0] ?? 0);
    this.#flatPatchDim = config.inChannels * temporalPatchSize * patchSize * patchSize;
    this.#embedDim = config.hiddenSize;

    const scale = 1 / Math.sqrt(this.#flatPatchDim);
    this.weight = random.uniform(-scale, scale, [
      config.hiddenSize,
      temporalPatchSize,
      patchSize,
      patchSize,
      config.inChannels,
    ]);
    this.bias = zeros([config.hiddenSize]);
  }

  forward(pixelValues: MxArray): MxArray {
    const [sequenceLength, patchDim] = pixelValues.shape;
    if (
      sequenceLength === undefined ||
      patchDim !== this.#flatPatchDim ||
      pixelValues.shape.length !== 2
    ) {
      throw new Error(
        `Qwen3_5VisionPatchEmbed.forward: expected [seq, ${this.#flatPatchDim}], got ${formatShape(pixelValues.shape)}.`,
      );
    }

    using castPixels =
      pixelValues.dtype === this.weight.dtype
        ? retainArray(pixelValues)
        : asType(pixelValues, this.weight.dtype);
    const projected = matmul(castPixels, this.transposedWeight());
    using unbiased = projected;
    return add(unbiased, this.bias);
  }

  private transposedWeight(): MxArray {
    if (this.#transposedWeight === null || this.#transposedWeightSource !== this.weight) {
      this.#transposedWeight?.free();
      using flattenedWeight = reshape(this.weight, [this.#embedDim, this.#flatPatchDim]);
      this.#transposedWeight = transpose(flattenedWeight);
      this.#transposedWeightSource = this.weight;
    }
    return this.#transposedWeight;
  }

  override [Symbol.dispose](): void {
    this.#transposedWeight?.free();
    this.#transposedWeight = null;
    this.#transposedWeightSource = null;
    super[Symbol.dispose]();
  }
}

class Qwen3_5VisionRotaryEmbedding {
  #invFreq: MxArray;

  constructor(dims: number, theta = 10_000) {
    const values = Array.from({ length: dims / 2 }, (_, index) => theta ** (-(2 * index) / dims));
    this.#invFreq = array(values, "float32");
  }

  forward(sequenceLength: number): MxArray {
    using positions = arange(0, sequenceLength, 1, "float32");
    using positionView = reshape(positions, [sequenceLength, 1]);
    using invFreqView = reshape(this.#invFreq, [1, this.#invFreq.shape[0] ?? 0]);
    return multiply(positionView, invFreqView);
  }

  [Symbol.dispose](): void {
    this.#invFreq.free();
  }
}

class Qwen3_5VisionMLP extends Module {
  linearFc1: Linear;
  linearFc2: Linear;

  constructor(config: Qwen3_5VisionConfig) {
    super();
    this.linearFc1 = new Linear(config.hiddenSize, config.intermediateSize, true);
    this.linearFc2 = new Linear(config.intermediateSize, config.hiddenSize, true);
  }

  forward(hiddenStates: MxArray): MxArray {
    using fc1 = this.linearFc1.forward(hiddenStates);
    using activated = gelu(fc1);
    return this.linearFc2.forward(activated);
  }
}

class Qwen3_5VisionAttention extends Module {
  qkv: Linear;
  proj: Linear;
  #hiddenSize: number;
  #numHeads: number;
  #headDim: number;

  constructor(config: Qwen3_5VisionConfig) {
    super();
    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numHeads;
    this.#headDim = config.hiddenSize / config.numHeads;
    this.qkv = new Linear(config.hiddenSize, config.hiddenSize * 3, true);
    this.proj = new Linear(config.hiddenSize, config.hiddenSize, true);
  }

  forward(hiddenStates: MxArray): MxArray {
    throw new Error(
      `Qwen3_5VisionAttention.forward: use run() with sequence metadata, got ${formatShape(hiddenStates.shape)}.`,
    );
  }

  run(
    hiddenStates: MxArray,
    sequenceLengths: readonly number[],
    cosEmbeddings: MxArray,
    sinEmbeddings: MxArray,
  ): MxArray {
    const [sequenceLength, hiddenSize] = hiddenStates.shape;
    if (
      sequenceLength === undefined ||
      hiddenSize !== this.#hiddenSize ||
      hiddenStates.shape.length !== 2
    ) {
      throw new Error(
        `Qwen3_5VisionAttention.run: expected [seq, ${this.#hiddenSize}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }

    using projected = this.qkv.forward(hiddenStates);
    using qkv = reshape(projected, [sequenceLength, 3, this.#numHeads, this.#headDim]);
    using queries = takeAxis1Slice(qkv, 0, "Qwen3_5VisionAttention.run");
    using keys = takeAxis1Slice(qkv, 1, "Qwen3_5VisionAttention.run");
    using values = takeAxis1Slice(qkv, 2, "Qwen3_5VisionAttention.run");
    const rotated = applyVisionRotaryPosEmb(queries, keys, cosEmbeddings, sinEmbeddings);
    try {
      const outputs: MxArray[] = [];
      let cursor = 0;
      try {
        for (const chunkLength of sequenceLengths) {
          using queryChunk = takeSequenceSlice(
            rotated.queries,
            cursor,
            chunkLength,
            "Qwen3_5VisionAttention.run",
          );
          using keyChunk = takeSequenceSlice(
            rotated.keys,
            cursor,
            chunkLength,
            "Qwen3_5VisionAttention.run",
          );
          using valueChunk = takeSequenceSlice(
            values,
            cursor,
            chunkLength,
            "Qwen3_5VisionAttention.run",
          );
          using transposedQueryChunk = transpose(queryChunk, [1, 0, 2]);
          using transposedKeyChunk = transpose(keyChunk, [1, 0, 2]);
          using transposedValueChunk = transpose(valueChunk, [1, 0, 2]);
          using queryBatch = expandDims(transposedQueryChunk, 0);
          using keyBatch = expandDims(transposedKeyChunk, 0);
          using valueBatch = expandDims(transposedValueChunk, 0);
          using attentionOutput = scaledDotProductAttention(queryBatch, keyBatch, valueBatch, {
            scale: this.#headDim ** -0.5,
          });
          using squeezedOutput = squeeze(attentionOutput, 0);
          using transposedOutput = transpose(squeezedOutput, [1, 0, 2]);
          outputs.push(reshape(transposedOutput, [chunkLength, this.#hiddenSize]));
          cursor += chunkLength;
        }

        using mergedOutput = concatenate(outputs, 0);
        return this.proj.forward(mergedOutput);
      } finally {
        for (const output of outputs) {
          output.free();
        }
      }
    } finally {
      rotated.queries.free();
      rotated.keys.free();
    }
  }
}

class Qwen3_5VisionBlock extends Module {
  norm1: LayerNorm;
  norm2: LayerNorm;
  attention: Qwen3_5VisionAttention;
  mlp: Qwen3_5VisionMLP;

  constructor(config: Qwen3_5VisionConfig) {
    super();
    this.norm1 = new LayerNorm(config.hiddenSize, 1e-6);
    this.norm2 = new LayerNorm(config.hiddenSize, 1e-6);
    this.attention = new Qwen3_5VisionAttention(config);
    this.mlp = new Qwen3_5VisionMLP(config);
  }

  forward(hiddenStates: MxArray): MxArray {
    throw new Error(
      `Qwen3_5VisionBlock.forward: use run() with sequence metadata, got ${formatShape(hiddenStates.shape)}.`,
    );
  }

  run(
    hiddenStates: MxArray,
    sequenceLengths: readonly number[],
    cosEmbeddings: MxArray,
    sinEmbeddings: MxArray,
  ): MxArray {
    using normalizedInputs = this.norm1.forward(hiddenStates);
    using attentionOutput = this.attention.run(
      normalizedInputs,
      sequenceLengths,
      cosEmbeddings,
      sinEmbeddings,
    );
    using attentionResidual = add(hiddenStates, attentionOutput);
    using normalizedResidual = this.norm2.forward(attentionResidual);
    using mlpOutput = this.mlp.forward(normalizedResidual);
    return add(attentionResidual, mlpOutput);
  }
}

class Qwen3_5VisionPatchMerger extends Module {
  norm: LayerNorm;
  linearFc1: Linear;
  linearFc2: Linear;
  #mergedHiddenSize: number;
  #usePostshuffleNorm: boolean;

  constructor(config: Qwen3_5VisionConfig, usePostshuffleNorm: boolean) {
    super();
    this.#mergedHiddenSize = config.hiddenSize * config.spatialMergeSize ** 2;
    this.#usePostshuffleNorm = usePostshuffleNorm;
    this.norm = new LayerNorm(
      usePostshuffleNorm ? this.#mergedHiddenSize : config.hiddenSize,
      1e-6,
    );
    this.linearFc1 = new Linear(this.#mergedHiddenSize, this.#mergedHiddenSize, true);
    this.linearFc2 = new Linear(this.#mergedHiddenSize, config.outHiddenSize, true);
  }

  forward(hiddenStates: MxArray): MxArray {
    if (this.#usePostshuffleNorm) {
      using groupedHidden = reshape(hiddenStates, [-1, this.#mergedHiddenSize]);
      using normalized = this.norm.forward(groupedHidden);
      using fc1 = this.linearFc1.forward(normalized);
      using activated = gelu(fc1);
      return this.linearFc2.forward(activated);
    }

    using normalized = this.norm.forward(hiddenStates);
    using groupedHidden = reshape(normalized, [-1, this.#mergedHiddenSize]);
    using fc1 = this.linearFc1.forward(groupedHidden);
    using activated = gelu(fc1);
    return this.linearFc2.forward(activated);
  }
}

/** Minimal Qwen 3.5 vision tower for image-conditioned prompts. */
export class Qwen3_5VisionModel extends Module {
  patchEmbed: Qwen3_5VisionPatchEmbed;
  posEmbed: Embedding;
  blocks: Qwen3_5VisionBlock[];
  merger: Qwen3_5VisionPatchMerger;
  #rotaryPosEmb: Qwen3_5VisionRotaryEmbedding;
  #spatialMergeSize: number;
  #gridSize: number;

  constructor(config: Qwen3_5VisionConfig) {
    super();
    if (config.deepstackVisualIndexes.length > 0) {
      throw new Error(
        "Qwen3_5VisionModel: DeepStack vision features are not implemented yet. Current Qwen3.6 dense checkpoints use an empty deepstack_visual_indexes list.",
      );
    }

    this.#spatialMergeSize = config.spatialMergeSize;
    this.#gridSize = Math.sqrt(config.numPositionEmbeddings);
    if (!Number.isInteger(this.#gridSize)) {
      throw new Error(
        `Qwen3_5VisionModel: numPositionEmbeddings ${config.numPositionEmbeddings} must be a perfect square.`,
      );
    }

    this.patchEmbed = new Qwen3_5VisionPatchEmbed(config);
    this.posEmbed = new Embedding(config.numPositionEmbeddings, config.hiddenSize);
    this.#rotaryPosEmb = new Qwen3_5VisionRotaryEmbedding(config.hiddenSize / config.numHeads / 2);
    this.blocks = Array.from({ length: config.depth }, () => new Qwen3_5VisionBlock(config));
    this.merger = new Qwen3_5VisionPatchMerger(config, false);
  }

  forward(pixelValues: MxArray, imageGridThw: MxArray): MxArray {
    const grids = gridThwList(imageGridThw, "Qwen3_5VisionModel.forward");
    const sequenceLengths = grids.map(([frames, height, width]) => frames * height * width);
    const totalTokens = sequenceLengths.reduce((sumValue, value) => sumValue + value, 0);
    const pixelSequenceLength = pixelValues.shape[0];
    if (pixelSequenceLength !== totalTokens || pixelValues.shape.length !== 2) {
      throw new Error(
        `Qwen3_5VisionModel.forward: pixelValues shape ${formatShape(pixelValues.shape)} does not match total grid tokens ${totalTokens}.`,
      );
    }

    using embeddedPatches = this.patchEmbed.forward(pixelValues);
    using positionalEmbeddings = this.fastPosEmbedInterpolate(grids);
    let hiddenStates = add(embeddedPatches, positionalEmbeddings);

    try {
      using rotaryEmbeddings = this.rotPosEmb(grids);
      using doubledRotary = concatenate([rotaryEmbeddings, rotaryEmbeddings], 1);
      using cosEmbeddings = cos(doubledRotary);
      using sinEmbeddings = sin(doubledRotary);

      for (const block of this.blocks) {
        using nextHiddenStates = block.run(
          hiddenStates,
          sequenceLengths,
          cosEmbeddings,
          sinEmbeddings,
        );
        hiddenStates.free();
        hiddenStates = retainArray(nextHiddenStates);
      }

      return this.merger.forward(hiddenStates);
    } finally {
      hiddenStates.free();
    }
  }

  override [Symbol.dispose](): void {
    this.#rotaryPosEmb[Symbol.dispose]();
    super[Symbol.dispose]();
  }

  private rotPosEmb(grids: readonly GridThw[]): MxArray {
    const maxHw = grids.reduce((maximum, [, height, width]) => Math.max(maximum, height, width), 0);
    using freqTable = this.#rotaryPosEmb.forward(maxHw);
    const perImageEmbeddings: MxArray[] = [];

    try {
      for (const grid of grids) {
        const geometry = gridGeometry(grid, this.#spatialMergeSize, "Qwen3_5VisionModel.rotPosEmb");
        const pairIndices = rotaryPairIndices(geometry, this.#spatialMergeSize);
        using indices = array(pairIndices, "int32");
        using pairEmbeddings = takeAxis(freqTable, indices, 0);
        perImageEmbeddings.push(
          flattenPairEmbeddings(pairEmbeddings, pairIndices.length, "Qwen3_5VisionModel.rotPosEmb"),
        );
      }

      return concatenate(perImageEmbeddings, 0);
    } finally {
      for (const embedding of perImageEmbeddings) {
        embedding.free();
      }
    }
  }

  private fastPosEmbedInterpolate(grids: readonly GridThw[]): MxArray {
    const mergedEmbeddings: MxArray[] = [];

    try {
      for (const grid of grids) {
        const [frames, height, width] = grid;
        const { indices, weights } = bilinearInterpolationTables(height, width, this.#gridSize);
        using indexTensor = array(indices, "int32");
        using weightTensor = array(weights, "float32");
        using patchPosEmbeddings = this.posEmbed.forward(indexTensor);
        using castWeights = asType(weightTensor, patchPosEmbeddings.dtype);
        using weightView = expandDims(castWeights, 2);
        using weightedEmbeddings = multiply(patchPosEmbeddings, weightView);
        using interpolatedEmbeddings = sum(weightedEmbeddings, 0);
        using repeatedEmbeddings = repeatFrames(interpolatedEmbeddings, frames);
        using reorderIndices = array(
          reorderedPatchIndices(
            grid,
            this.#spatialMergeSize,
            "Qwen3_5VisionModel.fastPosEmbedInterpolate",
          ),
          "int32",
        );
        mergedEmbeddings.push(takeAxis(repeatedEmbeddings, reorderIndices, 0));
      }

      return concatenate(mergedEmbeddings, 0);
    } finally {
      for (const embedding of mergedEmbeddings) {
        embedding.free();
      }
    }
  }
}
