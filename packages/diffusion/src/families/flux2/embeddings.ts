/**
 * FLUX.2 Klein timestep, guidance, and four-axis rotary embeddings.
 * @module
 */

import type { DType, MxArray } from "@mlxts/core";
import {
  add,
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

import type { Flux2KleinRopeAxes } from "./config";
import { assertFlux2Ids2d, sliceFlux2Axis } from "./tensor-utils";

/** Create the scalar embedding used by FLUX.2 timesteps and guidance. */
export function flux2TimestepEmbedding(
  timesteps: MxArray,
  dim = 256,
  options?: { maxPeriod?: number; timeFactor?: number; dtype?: DType },
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `flux2TimestepEmbedding: expected rank-1 timesteps, got ${formatShape(timesteps.shape)}.`,
    );
  }
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new Error("flux2TimestepEmbedding: dim must be a positive even integer.");
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

/** Bias-free two-layer MLP used by FLUX.2 scalar embedders. */
export class Flux2MLPEmbedder extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, hiddenDims: number) {
    super();
    this.linear1 = new Linear(inputDims, hiddenDims, false);
    this.linear2 = new Linear(hiddenDims, hiddenDims, false);
  }

  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = silu(hidden);
    return this.linear2.forward(activated);
  }
}

/** Combined timestep and optional guidance embedding used by FLUX.2. */
export class Flux2TimestepGuidanceEmbeddings extends Module {
  timestepEmbedder: Flux2MLPEmbedder;
  guidanceEmbedder: Flux2MLPEmbedder | null;
  #inputDims: number;

  constructor(inputDims: number, hiddenDims: number, guidanceEmbeds: boolean) {
    super();
    this.timestepEmbedder = new Flux2MLPEmbedder(inputDims, hiddenDims);
    this.guidanceEmbedder = guidanceEmbeds ? new Flux2MLPEmbedder(inputDims, hiddenDims) : null;
    this.#inputDims = inputDims;
  }

  forward(timestep: MxArray, guidance?: MxArray): MxArray {
    using timestepEmbedding = flux2TimestepEmbedding(timestep, this.#inputDims, {
      dtype: timestep.dtype,
      timeFactor: 1000,
    });
    using timestepVector = this.timestepEmbedder.forward(timestepEmbedding);
    if (guidance === undefined || this.guidanceEmbedder === null) {
      return retainArray(timestepVector);
    }
    using guidanceEmbedding = flux2TimestepEmbedding(guidance, this.#inputDims, {
      dtype: guidance.dtype,
      timeFactor: 1000,
    });
    using guidanceVector = this.guidanceEmbedder.forward(guidanceEmbedding);
    return add(timestepVector, guidanceVector);
  }
}

/** Four-axis RoPE matrix embedder used by FLUX.2 image and text ids. */
export class Flux2PosEmbed extends Module {
  #axesDims: Flux2KleinRopeAxes;
  #theta: number;
  #headDim: number;

  constructor(headDim: number, theta: number, axesDims: Flux2KleinRopeAxes) {
    super();
    const axesTotal = axesDims.reduce((sum, dim) => sum + dim, 0);
    if (axesTotal !== headDim) {
      throw new Error("Flux2PosEmbed: axesDimsRope sum must equal attentionHeadDim.");
    }
    for (const axisDim of axesDims) {
      if (axisDim % 2 !== 0) {
        throw new Error("Flux2PosEmbed: each axis dimension must be even.");
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
    assertFlux2Ids2d(ids, "Flux2PosEmbed.forward");
    const pieces: MxArray[] = [];
    try {
      for (let axis = 0; axis < this.#axesDims.length; axis += 1) {
        const axisDim = this.#axesDims[axis];
        if (axisDim === undefined) {
          throw new Error("Flux2PosEmbed.forward: missing RoPE axis dimension.");
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
    using positions = sliceFlux2Axis(ids, 1, axis, axis + 1);
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
