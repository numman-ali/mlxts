import type { DType, MxArray } from "@mlxts/core";
import {
  arange,
  asType,
  broadcastTo,
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

import type { QwenImageRopeAxes } from "./config";
import type { QwenImageRopeImageShape, QwenImageRopeImageShapes } from "./latents";

/** Create the sinusoidal scalar embedding used by Qwen-Image timesteps. */
export function qwenImageTimestepEmbedding(
  timesteps: MxArray,
  dim = 256,
  options?: { maxPeriod?: number; timeFactor?: number; dtype?: DType },
): MxArray {
  if (timesteps.shape.length !== 1) {
    throw new Error(
      `qwenImageTimestepEmbedding: expected rank-1 timesteps, got ${formatShape(timesteps.shape)}.`,
    );
  }
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new Error("qwenImageTimestepEmbedding: dim must be a positive even integer.");
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

/** Two-layer MLP used by Qwen-Image scalar conditioning. */
export class QwenImageTimestepEmbedder extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, hiddenDims: number) {
    super();
    this.linear1 = new Linear(inputDims, hiddenDims);
    this.linear2 = new Linear(hiddenDims, hiddenDims);
  }

  /** Project sinusoidal timestep embeddings into the Qwen-Image hidden size. */
  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = silu(hidden);
    return this.linear2.forward(activated);
  }
}

/** Diffusers-compatible Qwen-Image timestep projection surface. */
export class QwenImageTimestepProjEmbeddings extends Module {
  timestepEmbedder: QwenImageTimestepEmbedder;
  #useAdditionalTCond: boolean;

  constructor(hiddenSize: number, useAdditionalTCond: boolean) {
    super();
    this.timestepEmbedder = new QwenImageTimestepEmbedder(256, hiddenSize);
    this.#useAdditionalTCond = useAdditionalTCond;
  }

  forward(timestep: MxArray): MxArray {
    return this.embed(timestep, timestep.dtype);
  }

  /** Embed denoising timesteps for Qwen-Image block modulation. */
  embed(timestep: MxArray, dtype: DType, additionalTCond?: MxArray): MxArray {
    if (this.#useAdditionalTCond) {
      throw new Error("QwenImageTimestepProjEmbeddings.embed: additional_t_cond is unsupported.");
    }
    if (additionalTCond !== undefined) {
      throw new Error("QwenImageTimestepProjEmbeddings.embed: additionalTCond is not accepted.");
    }
    using projected = qwenImageTimestepEmbedding(timestep, 256, { dtype });
    return this.timestepEmbedder.forward(projected);
  }
}

function expectEvenAxisDims(axesDims: QwenImageRopeAxes, headDim: number): void {
  const total = axesDims.reduce((sum, dim) => sum + dim, 0);
  if (total !== headDim) {
    throw new Error("QwenImageRopeEmbedder: axesDims sum must equal headDim.");
  }
  for (const axisDim of axesDims) {
    if (axisDim % 2 !== 0) {
      throw new Error("QwenImageRopeEmbedder: each axis dimension must be even.");
    }
  }
}

function expectImageShape(imageShape: QwenImageRopeImageShape): void {
  for (const [index, value] of imageShape.entries()) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`QwenImageRopeEmbedder.embed: image shape axis ${index} must be positive.`);
    }
  }
}

function isImageShape(
  imageShapeOrShapes: QwenImageRopeImageShape | QwenImageRopeImageShapes,
): imageShapeOrShapes is QwenImageRopeImageShape {
  return typeof imageShapeOrShapes[0] === "number";
}

function normalizeImageShapes(
  imageShapeOrShapes: QwenImageRopeImageShape | QwenImageRopeImageShapes,
): QwenImageRopeImageShapes {
  const imageShapes = isImageShape(imageShapeOrShapes) ? [imageShapeOrShapes] : imageShapeOrShapes;
  if (imageShapes.length === 0) {
    throw new Error("QwenImageRopeEmbedder.embed: at least one image shape is required.");
  }
  return imageShapes;
}

function maxTextStartIndex(imageShapes: QwenImageRopeImageShapes): number {
  return imageShapes.reduce(
    (maxIndex, [, height, width]) =>
      Math.max(maxIndex, Math.floor(height / 2), Math.floor(width / 2)),
    0,
  );
}

function ropeMatricesForPositions(positions: MxArray, axisDim: number, theta: number): MxArray {
  using positionsFloat = asType(positions, "float32");
  using positionColumn = reshape(positionsFloat, [positions.shape[0] ?? 0, 1]);
  using indices = arange(0, axisDim, 2, "float32");
  using scaled = multiply(indices, -Math.log(theta) / axisDim);
  using omega = exp(scaled);
  using omegaRow = reshape(omega, [1, axisDim / 2]);
  using freqs = multiply(positionColumn, omegaRow);
  using cosine = cos(freqs);
  using sine = sin(freqs);
  using negativeSine = multiply(sine, -1);
  using row0 = stack([cosine, negativeSine], -1);
  using row1 = stack([sine, cosine], -1);
  return stack([row0, row1], -2);
}

function scaleRopePositions(length: number): MxArray {
  const negativeCount = length - Math.floor(length / 2);
  const positiveCount = Math.floor(length / 2);
  using negative = arange(-negativeCount, 0, 1, "float32");
  using positive = arange(0, positiveCount, 1, "float32");
  return concatenate([negative, positive], 0);
}

/** Qwen-Image 3-axis RoPE matrix builder for text and image token sequences. */
export class QwenImageRopeEmbedder extends Module {
  #axesDims: QwenImageRopeAxes;
  #theta: number;
  #headDim: number;

  constructor(headDim: number, theta: number, axesDims: QwenImageRopeAxes) {
    super();
    expectEvenAxisDims(axesDims, headDim);
    this.#axesDims = axesDims;
    this.#theta = theta;
    this.#headDim = headDim;
  }

  forward(_input: MxArray): MxArray {
    throw new Error("QwenImageRopeEmbedder.forward: use embed() with image shape metadata.");
  }

  /** Build combined `[text, image]` RoPE matrices for Qwen-Image joint attention. */
  embed(
    imageShapeOrShapes: QwenImageRopeImageShape | QwenImageRopeImageShapes,
    textLength: number,
    dtype: DType,
  ): MxArray {
    const imageShapes = normalizeImageShapes(imageShapeOrShapes);
    for (const imageShape of imageShapes) {
      expectImageShape(imageShape);
    }
    if (!Number.isInteger(textLength) || textLength <= 0) {
      throw new Error("QwenImageRopeEmbedder.embed: textLength must be positive.");
    }

    const imageRopes: MxArray[] = [];
    try {
      for (let index = 0; index < imageShapes.length; index += 1) {
        const imageShape = imageShapes[index];
        if (imageShape === undefined) {
          throw new Error("QwenImageRopeEmbedder.embed: missing image shape.");
        }
        const [frames, height, width] = imageShape;
        imageRopes.push(this.#imageRope(frames, height, width, index));
      }

      using text = this.#textRope(textLength, maxTextStartIndex(imageShapes));
      using image = concatenate(imageRopes, 0);
      using combined = concatenate([text, image], 0);
      using cast = combined.dtype === dtype ? retainArray(combined) : asType(combined, dtype);
      using batched = expandDims(cast, 0);
      return expandDims(batched, 0);
    } finally {
      for (const imageRope of imageRopes) {
        imageRope.free();
      }
    }
  }

  #textRope(textLength: number, startIndex: number): MxArray {
    using positions = arange(startIndex, startIndex + textLength, 1, "float32");
    const pieces: MxArray[] = [];
    try {
      for (const axisDim of this.#axesDims) {
        pieces.push(ropeMatricesForPositions(positions, axisDim, this.#theta));
      }
      return concatenate(pieces, 1);
    } finally {
      for (const piece of pieces) {
        piece.free();
      }
    }
  }

  #imageRope(frames: number, height: number, width: number, frameStartIndex: number): MxArray {
    const [frameDim, heightDim, widthDim] = this.#axesDims;
    using framePositions = arange(frameStartIndex, frameStartIndex + frames, 1, "float32");
    using heightPositions = scaleRopePositions(height);
    using widthPositions = scaleRopePositions(width);
    using frameRope = ropeMatricesForPositions(framePositions, frameDim, this.#theta);
    using heightRope = ropeMatricesForPositions(heightPositions, heightDim, this.#theta);
    using widthRope = ropeMatricesForPositions(widthPositions, widthDim, this.#theta);
    using frameGrid = broadcastTo(reshape(frameRope, [frames, 1, 1, frameDim / 2, 2, 2]), [
      frames,
      height,
      width,
      frameDim / 2,
      2,
      2,
    ]);
    using heightGrid = broadcastTo(reshape(heightRope, [1, height, 1, heightDim / 2, 2, 2]), [
      frames,
      height,
      width,
      heightDim / 2,
      2,
      2,
    ]);
    using widthGrid = broadcastTo(reshape(widthRope, [1, 1, width, widthDim / 2, 2, 2]), [
      frames,
      height,
      width,
      widthDim / 2,
      2,
      2,
    ]);
    using concatenated = concatenate([frameGrid, heightGrid, widthGrid], 3);
    return reshape(concatenated, [frames * height * width, this.#headDim / 2, 2, 2]);
  }

  get headDim(): number {
    return this.#headDim;
  }
}
