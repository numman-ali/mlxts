import type { DType, MxArray } from "@mlxts/core";
import {
  arange,
  asType,
  concatenate,
  cos,
  exp,
  formatShape,
  geluApprox,
  multiply,
  reshape,
  retainArray,
  sin,
} from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

const LTX_TIMESTEP_EMBEDDING_DIM = 256;
const SIN_COS_MAX_PERIOD = 10000;

export type LtxVideoAdaLayerNormSingleOutput = {
  modulation: MxArray;
  embeddedTimestep: MxArray;
};

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

/** Create the LTX/PixArt sinusoidal scalar embedding for FlowMatch timesteps. */
export function ltxVideoTimestepEmbedding(
  timesteps: MxArray,
  dim = LTX_TIMESTEP_EMBEDDING_DIM,
  dtype?: DType,
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `ltxVideoTimestepEmbedding: expected rank-1 timesteps, got ${formatShape(timesteps.shape)}.`,
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

/** Diffusers `TimestepEmbedding` used by LTX AdaLayerNormSingle. */
export class LtxVideoTimestepEmbedder extends Module {
  linear1: Linear;
  linear2: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.linear1 = new Linear(LTX_TIMESTEP_EMBEDDING_DIM, hiddenSize);
    this.linear2 = new Linear(hiddenSize, hiddenSize);
    this.#hiddenSize = hiddenSize;
  }

  /** Project sinusoidal timestep embeddings into hidden-size conditioning vectors. */
  embed(timesteps: MxArray, dtype: DType): MxArray {
    using timestepEmbedding = ltxVideoTimestepEmbedding(
      timesteps,
      LTX_TIMESTEP_EMBEDDING_DIM,
      dtype,
    );
    using hidden = this.linear1.forward(timestepEmbedding);
    using activated = silu(hidden);
    using output = this.linear2.forward(activated);
    const [, channels] = output.shape;
    if (channels !== this.#hiddenSize) {
      throw new Error("LtxVideoTimestepEmbedder.forward: hidden size mismatch.");
    }
    return retainArray(output);
  }

  forward(timesteps: MxArray): MxArray {
    return this.embed(timesteps, timesteps.dtype);
  }
}

/** PixArt timestep embedding wrapper used by Diffusers LTX. */
export class LtxVideoCombinedTimestepSizeEmbeddings extends Module {
  timestepEmbedder: LtxVideoTimestepEmbedder;

  constructor(hiddenSize: number) {
    super();
    this.timestepEmbedder = new LtxVideoTimestepEmbedder(hiddenSize);
  }

  /** Return the timestep embedding without additional resolution or aspect-ratio conditions. */
  embed(timesteps: MxArray, dtype: DType): MxArray {
    return this.timestepEmbedder.embed(timesteps, dtype);
  }

  forward(timesteps: MxArray): MxArray {
    return this.embed(timesteps, timesteps.dtype);
  }
}

/** Diffusers `AdaLayerNormSingle` modulation used by LTX transformer blocks. */
export class LtxVideoAdaLayerNormSingle extends Module {
  emb: LtxVideoCombinedTimestepSizeEmbeddings;
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.emb = new LtxVideoCombinedTimestepSizeEmbeddings(hiddenSize);
    this.linear = new Linear(hiddenSize, hiddenSize * 6);
    this.#hiddenSize = hiddenSize;
  }

  /** Return block modulation and final embedded timestep tensors. */
  embed(timesteps: MxArray, dtype: DType): LtxVideoAdaLayerNormSingleOutput {
    const [batch] = timesteps.shape;
    if (timesteps.shape.length !== 1 || batch === undefined) {
      throw new Error(
        `LtxVideoAdaLayerNormSingle.embed: timestep must be rank 1, got ${formatShape(
          timesteps.shape,
        )}.`,
      );
    }
    using embeddedTimestepFlat = this.emb.embed(timesteps, dtype);
    using activated = silu(embeddedTimestepFlat);
    using modulationFlat = this.linear.forward(activated);
    using modulation = reshape(modulationFlat, [batch, 6, this.#hiddenSize]);
    using embeddedTimestep = reshape(embeddedTimestepFlat, [batch, 1, this.#hiddenSize]);
    return {
      modulation: retainArray(modulation),
      embeddedTimestep: retainArray(embeddedTimestep),
    };
  }

  forward(timesteps: MxArray): MxArray {
    const output = this.embed(timesteps, timesteps.dtype);
    try {
      return retainArray(output.modulation);
    } finally {
      disposeLtxVideoAdaLayerNormSingleOutput(output);
    }
  }
}

/** Dispose tensors returned by `LtxVideoAdaLayerNormSingle.embed`. */
export function disposeLtxVideoAdaLayerNormSingleOutput(
  output: LtxVideoAdaLayerNormSingleOutput,
): void {
  output.modulation.free();
  output.embeddedTimestep.free();
}

/** Diffusers PixArt caption projection used by LTX text conditioning. */
export class LtxVideoCaptionProjection extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, hiddenSize: number) {
    super();
    this.linear1 = new Linear(inputDims, hiddenSize);
    this.linear2 = new Linear(hiddenSize, hiddenSize);
  }

  /** Project T5 caption embeddings into LTX transformer hidden size. */
  forward(caption: MxArray): MxArray {
    using hidden = this.linear1.forward(caption);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}
