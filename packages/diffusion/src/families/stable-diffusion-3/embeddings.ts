import type { DType, MxArray } from "@mlxts/core";
import {
  add,
  arange,
  array,
  asType,
  concatenate,
  cos,
  exp,
  formatShape,
  multiply,
  reshape,
  retainArray,
  sin,
} from "@mlxts/core";
import { Conv2d, Linear, Module, silu } from "@mlxts/nn";

import { assertImage4d } from "./tensor-utils";

const SD3_TIMESTEP_EMBEDDING_DIM = 256;
const SIN_COS_MAX_PERIOD = 10000;

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectDivisible(value: number, divisor: number, name: string): void {
  if (value % divisor !== 0) {
    throw new Error(`${name} must be divisible by ${divisor}.`);
  }
}

function positionCacheKey(height: number, width: number, dtype: DType): string {
  return `${height}x${width}:${dtype}`;
}

function oneAxisEmbedding(
  position: number,
  axisDim: number,
  output: Float32Array,
  offset: number,
): void {
  const halfDim = axisDim / 2;
  for (let index = 0; index < halfDim; index += 1) {
    const omega = 1 / SIN_COS_MAX_PERIOD ** (index / halfDim);
    const value = position * omega;
    output[offset + index] = Math.sin(value);
    output[offset + halfDim + index] = Math.cos(value);
  }
}

function make2dSincosPositionEmbedding(options: {
  hiddenSize: number;
  gridHeight: number;
  gridWidth: number;
  posEmbedMaxSize: number;
  baseSize: number;
}): Float32Array {
  const { hiddenSize, gridHeight, gridWidth, posEmbedMaxSize, baseSize } = options;
  expectDivisible(hiddenSize, 4, "hiddenSize");
  const axisDim = hiddenSize / 2;
  const values = new Float32Array(gridHeight * gridWidth * hiddenSize);
  const top = Math.floor((posEmbedMaxSize - gridHeight) / 2);
  const left = Math.floor((posEmbedMaxSize - gridWidth) / 2);
  const scale = posEmbedMaxSize / baseSize;
  for (let row = 0; row < gridHeight; row += 1) {
    for (let column = 0; column < gridWidth; column += 1) {
      const tokenOffset = (row * gridWidth + column) * hiddenSize;
      oneAxisEmbedding((left + column) / scale, axisDim, values, tokenOffset);
      oneAxisEmbedding((top + row) / scale, axisDim, values, tokenOffset + axisDim);
    }
  }
  return values;
}

/** Create the SD3 sinusoidal scalar embedding for FlowMatch timesteps. */
export function stableDiffusion3TimestepEmbedding(
  timesteps: MxArray,
  dim = SD3_TIMESTEP_EMBEDDING_DIM,
  dtype?: DType,
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `stableDiffusion3TimestepEmbedding: expected rank-1 timesteps, got ${formatShape(
        timesteps.shape,
      )}.`,
    );
  }
  expectPositiveInteger(dim, "dim");
  expectDivisible(dim, 2, "dim");
  const halfDim = dim / 2;
  using frequencyIndex = arange(0, halfDim, 1, "float32");
  using logFrequencies = multiply(frequencyIndex, -Math.log(SIN_COS_MAX_PERIOD) / halfDim);
  using frequencies = exp(logFrequencies);
  using timestepColumn = reshape(timesteps, [timesteps.shape[0] ?? 0, 1]);
  using frequencyRow = reshape(frequencies, [1, halfDim]);
  using args = multiply(timestepColumn, frequencyRow);
  using sine = sin(args);
  using cosine = cos(args);
  using embedding = concatenate([cosine, sine], -1);
  if (dtype === undefined || embedding.dtype === dtype) {
    return retainArray(embedding);
  }
  return asType(embedding, dtype);
}

/** Patch embedder used by Diffusers `SD3Transformer2DModel` over NHWC latents. */
export class StableDiffusion3PatchEmbed extends Module {
  projection: Conv2d;
  #hiddenSize: number;
  #patchSize: number;
  #posEmbedMaxSize: number;
  #baseSize: number;
  #positionCache = new Map<string, MxArray>();

  constructor(options: {
    sampleSize: number;
    patchSize: number;
    inChannels: number;
    hiddenSize: number;
    posEmbedMaxSize: number;
  }) {
    super();
    expectPositiveInteger(options.sampleSize, "sampleSize");
    expectPositiveInteger(options.patchSize, "patchSize");
    expectPositiveInteger(options.inChannels, "inChannels");
    expectPositiveInteger(options.hiddenSize, "hiddenSize");
    expectPositiveInteger(options.posEmbedMaxSize, "posEmbedMaxSize");
    expectDivisible(options.sampleSize, options.patchSize, "sampleSize");
    expectDivisible(options.hiddenSize, 4, "hiddenSize");
    this.projection = new Conv2d(
      options.inChannels,
      options.hiddenSize,
      options.patchSize,
      options.patchSize,
    );
    this.#hiddenSize = options.hiddenSize;
    this.#patchSize = options.patchSize;
    this.#posEmbedMaxSize = options.posEmbedMaxSize;
    this.#baseSize = options.sampleSize / options.patchSize;
  }

  forward(latents: MxArray): MxArray {
    const shape = assertImage4d(latents, "StableDiffusion3PatchEmbed.forward");
    expectDivisible(shape.height, this.#patchSize, "latent height");
    expectDivisible(shape.width, this.#patchSize, "latent width");
    const gridHeight = shape.height / this.#patchSize;
    const gridWidth = shape.width / this.#patchSize;
    if (gridHeight > this.#posEmbedMaxSize || gridWidth > this.#posEmbedMaxSize) {
      throw new Error(
        "StableDiffusion3PatchEmbed.forward: latent patch grid exceeds posEmbedMaxSize.",
      );
    }

    using projected = this.projection.forward(latents);
    using sequence = reshape(projected, [shape.batch, gridHeight * gridWidth, this.#hiddenSize]);
    using position = this.#positionEmbedding(gridHeight, gridWidth, latents.dtype);
    return add(sequence, position);
  }

  #positionEmbedding(gridHeight: number, gridWidth: number, dtype: DType): MxArray {
    const key = positionCacheKey(gridHeight, gridWidth, dtype);
    const cached = this.#positionCache.get(key);
    if (cached !== undefined) {
      return retainArray(cached);
    }

    const values = make2dSincosPositionEmbedding({
      hiddenSize: this.#hiddenSize,
      gridHeight,
      gridWidth,
      posEmbedMaxSize: this.#posEmbedMaxSize,
      baseSize: this.#baseSize,
    });
    const flat = array(values, "float32");
    try {
      const shaped = reshape(flat, [1, gridHeight * gridWidth, this.#hiddenSize]);
      if (dtype === "float32") {
        this.#positionCache.set(key, shaped);
        return retainArray(shaped);
      }
      try {
        const cast = asType(shaped, dtype);
        this.#positionCache.set(key, cast);
        return retainArray(cast);
      } finally {
        shaped.free();
      }
    } finally {
      flat.free();
    }
  }

  override [Symbol.dispose](): void {
    for (const value of this.#positionCache.values()) {
      value.free();
    }
    this.#positionCache.clear();
    super[Symbol.dispose]();
  }
}

/** SD3 two-layer caption projection for pooled CLIP/T5 conditioning. */
export class StableDiffusion3PixArtTextProjection extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, hiddenSize: number) {
    super();
    this.linear1 = new Linear(inputDims, hiddenSize);
    this.linear2 = new Linear(hiddenSize, hiddenSize);
  }

  forward(caption: MxArray): MxArray {
    using hidden = this.linear1.forward(caption);
    using activated = silu(hidden);
    return this.linear2.forward(activated);
  }
}

/** SD3 timestep-plus-pooled-text conditioning embedder. */
export class StableDiffusion3TimestepTextEmbeddings extends Module {
  timestepLinear1: Linear;
  timestepLinear2: Linear;
  textEmbedder: StableDiffusion3PixArtTextProjection;
  #hiddenSize: number;

  constructor(hiddenSize: number, pooledProjectionDim: number) {
    super();
    this.timestepLinear1 = new Linear(SD3_TIMESTEP_EMBEDDING_DIM, hiddenSize);
    this.timestepLinear2 = new Linear(hiddenSize, hiddenSize);
    this.textEmbedder = new StableDiffusion3PixArtTextProjection(pooledProjectionDim, hiddenSize);
    this.#hiddenSize = hiddenSize;
  }

  forward(timestep: MxArray, pooledProjections: MxArray): MxArray {
    const [batch, pooledChannels] = pooledProjections.shape;
    if (
      pooledProjections.shape.length !== 2 ||
      batch === undefined ||
      pooledChannels === undefined
    ) {
      throw new Error(
        `StableDiffusion3TimestepTextEmbeddings.forward: pooledProjections must be rank 2, got ${formatShape(
          pooledProjections.shape,
        )}.`,
      );
    }
    if (timestep.shape.length !== 1 || timestep.shape[0] !== batch) {
      throw new Error(
        `StableDiffusion3TimestepTextEmbeddings.forward: timestep must have shape [${batch}], got ${formatShape(
          timestep.shape,
        )}.`,
      );
    }
    using timestepEmbedding = stableDiffusion3TimestepEmbedding(
      timestep,
      SD3_TIMESTEP_EMBEDDING_DIM,
      pooledProjections.dtype,
    );
    using timestepHidden = this.timestepLinear1.forward(timestepEmbedding);
    using activated = silu(timestepHidden);
    using timeVector = this.timestepLinear2.forward(activated);
    using textVector = this.textEmbedder.forward(pooledProjections);
    const [, timeChannels] = timeVector.shape;
    if (timeChannels !== this.#hiddenSize) {
      throw new Error("StableDiffusion3TimestepTextEmbeddings.forward: hidden size mismatch.");
    }
    return add(timeVector, textVector);
  }
}
