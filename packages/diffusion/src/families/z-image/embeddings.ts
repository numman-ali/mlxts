/**
 * Z-Image scalar, caption, and rotary embeddings.
 * @module
 */

import type { DType, MxArray } from "@mlxts/core";
import {
  arange,
  asType,
  concatenate,
  cos,
  exp,
  expandDims,
  formatShape,
  multiply,
  reshape,
  retainArray,
  sin,
  stack,
} from "@mlxts/core";
import { Linear, Module, RMSNorm, silu } from "@mlxts/nn";

import type { ZImageRopeAxes, ZImageRopeAxisLengths } from "./config";
import { assertIds2d, sliceAxis } from "./tensor-utils";

const Z_IMAGE_TIMESTEP_EMBEDDING_DIM = 256;
const Z_IMAGE_TIMESTEP_MLP_HIDDEN = 1024;

/** Create the sinusoidal scalar embedding used by Z-Image timesteps. */
export function zImageTimestepEmbedding(
  timesteps: MxArray,
  dim = Z_IMAGE_TIMESTEP_EMBEDDING_DIM,
  options?: { maxPeriod?: number; dtype?: DType },
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `zImageTimestepEmbedding: expected rank-1 timesteps, got ${formatShape(timesteps.shape)}.`,
    );
  }
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new Error("zImageTimestepEmbedding: dim must be a positive even integer.");
  }

  const maxPeriod = options?.maxPeriod ?? 10000;
  const halfDim = dim / 2;
  using frequencyIndex = arange(0, halfDim, 1, "float32");
  using logFrequencies = multiply(frequencyIndex, -Math.log(maxPeriod) / halfDim);
  using frequencies = exp(logFrequencies);
  using timestepColumn = reshape(timesteps, [timesteps.shape[0] ?? 0, 1]);
  using frequencyRow = reshape(frequencies, [1, halfDim]);
  using args = multiply(timestepColumn, frequencyRow);
  using cosine = cos(args);
  using sine = sin(args);
  using embedding = concatenate([cosine, sine], -1);
  if (options?.dtype === undefined || embedding.dtype === options.dtype) {
    return retainArray(embedding);
  }
  return asType(embedding, options.dtype);
}

/** Diffusers-compatible Z-Image timestep MLP. */
export class ZImageTimestepEmbedder extends Module {
  linear1: Linear;
  linear2: Linear;
  #outputDims: number;

  constructor(outputDims: number, hiddenDims = Z_IMAGE_TIMESTEP_MLP_HIDDEN) {
    super();
    this.linear1 = new Linear(Z_IMAGE_TIMESTEP_EMBEDDING_DIM, hiddenDims);
    this.linear2 = new Linear(hiddenDims, outputDims);
    this.#outputDims = outputDims;
  }

  forward(timesteps: MxArray): MxArray {
    return this.embed(timesteps, timesteps.dtype);
  }

  /** Project normalized timesteps into the AdaLN conditioning dimension. */
  embed(timesteps: MxArray, dtype: DType): MxArray {
    using embedding = zImageTimestepEmbedding(timesteps, Z_IMAGE_TIMESTEP_EMBEDDING_DIM, {
      dtype,
    });
    using hidden = this.linear1.forward(embedding);
    using activated = silu(hidden);
    return this.linear2.forward(activated);
  }

  get outputDims(): number {
    return this.#outputDims;
  }
}

/** Caption feature normalization and projection used before Z-Image context refinement. */
export class ZImageCaptionEmbedder extends Module {
  norm: RMSNorm;
  projection: Linear;
  #inputDims: number;

  constructor(inputDims: number, hiddenSize: number, eps: number) {
    super();
    this.norm = new RMSNorm(inputDims, eps);
    this.projection = new Linear(inputDims, hiddenSize);
    this.#inputDims = inputDims;
  }

  /** Project Qwen caption features into Z-Image hidden states. */
  forward(captionFeatures: MxArray): MxArray {
    const lastDimension = captionFeatures.shape[captionFeatures.shape.length - 1];
    if (lastDimension !== this.#inputDims) {
      throw new Error(
        `ZImageCaptionEmbedder.forward: expected last dimension ${this.#inputDims}, got ${
          lastDimension ?? "undefined"
        } for shape ${formatShape(captionFeatures.shape)}.`,
      );
    }
    using normalized = this.norm.forward(captionFeatures);
    return this.projection.forward(normalized);
  }
}

/** N-dimensional Z-Image RoPE matrix embedder for caption and latent ids. */
export class ZImageRopeEmbedder extends Module {
  #axesDims: ZImageRopeAxes;
  #axesLens: ZImageRopeAxisLengths;
  #theta: number;
  #headDim: number;

  constructor(
    headDim: number,
    theta: number,
    axesDims: ZImageRopeAxes,
    axesLens: ZImageRopeAxisLengths,
  ) {
    super();
    const axesTotal = axesDims.reduce((sum, dim) => sum + dim, 0);
    if (axesTotal !== headDim) {
      throw new Error("ZImageRopeEmbedder: axesDims sum must equal headDim.");
    }
    for (const axisDim of axesDims) {
      if (axisDim % 2 !== 0) {
        throw new Error("ZImageRopeEmbedder: each axis dimension must be even.");
      }
    }
    this.#axesDims = axesDims;
    this.#axesLens = axesLens;
    this.#theta = theta;
    this.#headDim = headDim;
  }

  forward(ids: MxArray): MxArray {
    return this.embed(ids, ids.dtype);
  }

  /** Build RoPE matrices with shape `[1, 1, length, headDim / 2, 2, 2]`. */
  embed(ids: MxArray, dtype: DType): MxArray {
    const { length } = assertIds2d(ids, "ZImageRopeEmbedder.forward");
    for (let axis = 0; axis < this.#axesLens.length; axis += 1) {
      const axisLength = this.#axesLens[axis];
      if (axisLength === undefined) {
        throw new Error("ZImageRopeEmbedder.forward: missing RoPE axis length.");
      }
    }

    const pieces: MxArray[] = [];
    try {
      for (let axis = 0; axis < this.#axesDims.length; axis += 1) {
        const axisDim = this.#axesDims[axis];
        if (axisDim === undefined) {
          throw new Error("ZImageRopeEmbedder.forward: missing RoPE axis dimension.");
        }
        pieces.push(this.#ropeAxis(ids, axis, axisDim));
      }
      using concatenated = concatenate(pieces, 1);
      using cast =
        concatenated.dtype === dtype ? retainArray(concatenated) : asType(concatenated, dtype);
      using batched = expandDims(cast, 0);
      const output = expandDims(batched, 0);
      if (output.shape[2] !== length) {
        output.free();
        throw new Error("ZImageRopeEmbedder.forward: embedded length mismatch.");
      }
      return output;
    } finally {
      for (const piece of pieces) {
        piece.free();
      }
    }
  }

  #ropeAxis(ids: MxArray, axis: number, axisDim: number): MxArray {
    using positions = sliceAxis(ids, 1, axis, axis + 1);
    using positionsFloat = asType(positions, "float32");
    using indices = arange(0, axisDim, 2, "float32");
    using scaled = multiply(indices, -Math.log(this.#theta) / axisDim);
    using omega = exp(scaled);
    using omegaRow = reshape(omega, [1, axisDim / 2]);
    using freqs = multiply(positionsFloat, omegaRow);
    using cosine = cos(freqs);
    using sine = sin(freqs);
    using negativeSine = multiply(sine, -1);
    using row0 = stack([cosine, negativeSine], -1);
    using row1 = stack([sine, cosine], -1);
    return stack([row0, row1], -2);
  }

  get headDim(): number {
    return this.#headDim;
  }
}
