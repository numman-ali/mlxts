/**
 * FLUX.1 timestep and rotary-position embeddings.
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
import { Linear, Module, silu } from "@mlxts/nn";

import type { FluxRopeAxes } from "./config";
import { assertIds2d, sliceAxis } from "./tensor-utils";

/** Create the sinusoidal scalar embedding used for FLUX timesteps and guidance. */
export function fluxTimestepEmbedding(
  timesteps: MxArray,
  dim = 256,
  options?: { maxPeriod?: number; timeFactor?: number; dtype?: DType },
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `fluxTimestepEmbedding: expected rank-1 timesteps, got ${formatShape(timesteps.shape)}.`,
    );
  }
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new Error("fluxTimestepEmbedding: dim must be a positive even integer.");
  }

  const maxPeriod = options?.maxPeriod ?? 10000;
  const timeFactor = options?.timeFactor ?? 1000;
  const halfDim = dim / 2;
  using frequencyIndex = arange(0, halfDim, 1, "float32");
  using logFrequencies = multiply(frequencyIndex, -Math.log(maxPeriod) / halfDim);
  using frequencies = exp(logFrequencies);
  using scaledTimesteps = multiply(timesteps, timeFactor);
  using timestepColumn = reshape(scaledTimesteps, [timesteps.shape[0] ?? 0, 1]);
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

/** Two-layer MLP used by FLUX scalar/vector conditioning embedders. */
export class FluxMLPEmbedder extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, hiddenDims: number) {
    super();
    this.linear1 = new Linear(inputDims, hiddenDims);
    this.linear2 = new Linear(hiddenDims, hiddenDims);
  }

  /** Project a conditioning vector into the FLUX hidden size. */
  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = silu(hidden);
    return this.linear2.forward(activated);
  }
}

/** N-dimensional RoPE matrix embedder used by FLUX image/text ids. */
export class FluxEmbedND extends Module {
  #axesDims: FluxRopeAxes;
  #theta: number;
  #headDim: number;

  constructor(headDim: number, theta: number, axesDims: FluxRopeAxes) {
    super();
    const axesTotal = axesDims.reduce((sum, dim) => sum + dim, 0);
    if (axesTotal !== headDim) {
      throw new Error("FluxEmbedND: axesDims sum must equal headDim.");
    }
    for (const axisDim of axesDims) {
      if (axisDim % 2 !== 0) {
        throw new Error("FluxEmbedND: each axis dimension must be even.");
      }
    }
    this.#axesDims = axesDims;
    this.#theta = theta;
    this.#headDim = headDim;
  }

  forward(ids: MxArray): MxArray {
    return this.embed(ids, ids.dtype);
  }

  /** Build RoPE matrices with shape `[1, 1, length, headDim / 2, 2, 2]`. */
  embed(ids: MxArray, dtype: DType): MxArray {
    assertIds2d(ids, "FluxEmbedND.forward");
    const pieces: MxArray[] = [];
    try {
      for (let axis = 0; axis < this.#axesDims.length; axis += 1) {
        const axisDim = this.#axesDims[axis];
        if (axisDim === undefined) {
          throw new Error("FluxEmbedND.forward: missing RoPE axis dimension.");
        }
        pieces.push(this.#ropeAxis(ids, axis, axisDim));
      }
      using concatenated = concatenate(pieces, 1);
      using cast =
        concatenated.dtype === dtype ? retainArray(concatenated) : asType(concatenated, dtype);
      using batched = expandDims(cast, 0);
      return expandDims(batched, 0);
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
