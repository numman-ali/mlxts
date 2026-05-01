import { MxArray } from "@mlxts/core";

import type { Ltx2RopeType } from "./config";

export type Ltx2RotaryEmbeddingOptions = {
  coords: MxArray;
  dim: number;
  modality: "video" | "audio";
  ropeType?: Ltx2RopeType;
  theta?: number;
  baseNumFrames?: number;
  baseHeight?: number;
  baseWidth?: number;
  numAttentionHeads?: number;
};

export type LtxRotaryEmbeddings = {
  cos: MxArray;
  sin: MxArray;
};

type Ltx2CoordGrid = {
  batchSize: number;
  tokens: number;
  positionDims: number;
  values: Float32Array;
};

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectPositiveNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}

function resolvePositiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  expectPositiveInteger(resolved, name);
  return resolved;
}

function resolvePositiveNumber(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  expectPositiveNumber(resolved, name);
  return resolved;
}

function linspaceUnit(index: number, count: number): number {
  if (count === 1) {
    return 0;
  }
  return index / (count - 1);
}

function ltx2CoordGrid(options: Ltx2RotaryEmbeddingOptions): Ltx2CoordGrid {
  const { batchSize, positionDims, tokens } = expectLtx2CoordShape(options);
  const basePositions = ltx2BasePositions(options);
  const typed = options.coords.toTypedArray();
  const values = new Float32Array(batchSize * tokens * positionDims);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let token = 0; token < tokens; token += 1) {
      for (let axis = 0; axis < positionDims; axis += 1) {
        const inputOffset = ((batch * positionDims + axis) * tokens + token) * 2;
        const start = Number(typed[inputOffset] ?? 0);
        const end = Number(typed[inputOffset + 1] ?? 0);
        const base = basePositions[axis] ?? 1;
        values[(batch * tokens + token) * positionDims + axis] = (start + end) / 2 / base;
      }
    }
  }
  return { batchSize, tokens, positionDims, values };
}

function expectLtx2CoordShape(options: Ltx2RotaryEmbeddingOptions): {
  batchSize: number;
  positionDims: number;
  tokens: number;
} {
  const [batchSize, positionDims, tokens, bounds] = options.coords.shape;
  if (
    options.coords.shape.length !== 4 ||
    batchSize === undefined ||
    positionDims === undefined ||
    tokens === undefined ||
    bounds !== 2
  ) {
    throw new Error("LTX-2 RoPE coordinates must have shape [batch, axes, tokens, 2].");
  }
  const expectedPositionDims = options.modality === "video" ? 3 : 1;
  if (positionDims !== expectedPositionDims) {
    throw new Error(
      `LTX-2 ${options.modality} RoPE requires ${expectedPositionDims} coordinate axes.`,
    );
  }
  return { batchSize, positionDims, tokens };
}

function ltx2BasePositions(options: Ltx2RotaryEmbeddingOptions): readonly number[] {
  if (options.modality === "audio") {
    return [options.baseNumFrames ?? 20];
  }
  return [options.baseNumFrames ?? 20, options.baseHeight ?? 2048, options.baseWidth ?? 2048];
}

function resolveLtx2RopeType(ropeType: string | undefined): Ltx2RopeType {
  const resolved = ropeType ?? "interleaved";
  if (resolved !== "interleaved" && resolved !== "split") {
    throw new Error("LTX-2 ropeType must be 'interleaved' or 'split'.");
  }
  return resolved;
}

/** Precompute LTX-2 cosine/sine RoPE tensors for video or audio coordinates. */
export function createLtx2RotaryEmbeddings(
  options: Ltx2RotaryEmbeddingOptions,
): LtxRotaryEmbeddings {
  expectPositiveInteger(options.dim, "dim");
  const grid = ltx2CoordGrid(options);
  const ropeType = resolveLtx2RopeType(options.ropeType);
  const numRopeElems = grid.positionDims * 2;
  const frequencyCount = Math.floor(options.dim / numRopeElems);
  if (frequencyCount <= 0) {
    throw new Error("dim is too small for LTX-2 RoPE axes.");
  }
  const theta = resolvePositiveNumber(options.theta, 10000, "theta");
  if (ropeType === "split") {
    return createLtx2SplitRotaryEmbeddings(options, grid, frequencyCount, theta);
  }
  return createLtx2InterleavedRotaryEmbeddings(options, grid, frequencyCount, theta);
}

function createLtx2InterleavedRotaryEmbeddings(
  options: Ltx2RotaryEmbeddingOptions,
  grid: Ltx2CoordGrid,
  frequencyCount: number,
  theta: number,
): LtxRotaryEmbeddings {
  const padding = options.dim % (grid.positionDims * 2);
  const cosValues = new Float32Array(grid.batchSize * grid.tokens * options.dim);
  const sinValues = new Float32Array(cosValues.length);
  for (let batch = 0; batch < grid.batchSize; batch += 1) {
    for (let token = 0; token < grid.tokens; token += 1) {
      writeLtx2InterleavedToken({ cosValues, sinValues }, options, grid, {
        batch,
        token,
        frequencyCount,
        padding,
        theta,
      });
    }
  }
  return {
    cos: MxArray.fromData(cosValues, [grid.batchSize, grid.tokens, options.dim], "float32"),
    sin: MxArray.fromData(sinValues, [grid.batchSize, grid.tokens, options.dim], "float32"),
  };
}

function writeLtx2InterleavedToken(
  output: { cosValues: Float32Array; sinValues: Float32Array },
  options: Ltx2RotaryEmbeddingOptions,
  grid: Ltx2CoordGrid,
  tokenShape: {
    batch: number;
    token: number;
    frequencyCount: number;
    padding: number;
    theta: number;
  },
): void {
  const outputOffset = (tokenShape.batch * grid.tokens + tokenShape.token) * options.dim;
  for (let index = 0; index < tokenShape.padding; index += 1) {
    output.cosValues[outputOffset + index] = 1;
  }
  for (let frequencyIndex = 0; frequencyIndex < tokenShape.frequencyCount; frequencyIndex += 1) {
    const frequency =
      tokenShape.theta ** linspaceUnit(frequencyIndex, tokenShape.frequencyCount) * (Math.PI / 2);
    for (let axis = 0; axis < grid.positionDims; axis += 1) {
      const coord =
        grid.values[
          (tokenShape.batch * grid.tokens + tokenShape.token) * grid.positionDims + axis
        ] ?? 0;
      const angle = frequency * (coord * 2 - 1);
      const offset =
        outputOffset + tokenShape.padding + (frequencyIndex * grid.positionDims + axis) * 2;
      const cosValue = Math.cos(angle);
      const sinValue = Math.sin(angle);
      output.cosValues[offset] = cosValue;
      output.cosValues[offset + 1] = cosValue;
      output.sinValues[offset] = sinValue;
      output.sinValues[offset + 1] = sinValue;
    }
  }
}

function createLtx2SplitRotaryEmbeddings(
  options: Ltx2RotaryEmbeddingOptions,
  grid: Ltx2CoordGrid,
  frequencyCount: number,
  theta: number,
): LtxRotaryEmbeddings {
  const numAttentionHeads = resolvePositiveInteger(
    options.numAttentionHeads,
    32,
    "numAttentionHeads",
  );
  const expectedFreqs = Math.floor(options.dim / 2);
  if (expectedFreqs % numAttentionHeads !== 0) {
    throw new Error("LTX-2 split RoPE dim must divide evenly across attention heads.");
  }
  const padding = expectedFreqs - frequencyCount * grid.positionDims;
  if (padding < 0) {
    throw new Error("LTX-2 split RoPE frequency count exceeds dim / 2.");
  }
  const headDim = expectedFreqs / numAttentionHeads;
  const cosValues = new Float32Array(grid.batchSize * numAttentionHeads * grid.tokens * headDim);
  const sinValues = new Float32Array(cosValues.length);
  for (let batch = 0; batch < grid.batchSize; batch += 1) {
    for (let token = 0; token < grid.tokens; token += 1) {
      writeLtx2SplitToken(
        { cosValues, sinValues },
        { batch, token, numAttentionHeads, headDim, expectedFreqs, padding },
        grid,
        frequencyCount,
        theta,
      );
    }
  }
  return {
    cos: MxArray.fromData(
      cosValues,
      [grid.batchSize, numAttentionHeads, grid.tokens, headDim],
      "float32",
    ),
    sin: MxArray.fromData(
      sinValues,
      [grid.batchSize, numAttentionHeads, grid.tokens, headDim],
      "float32",
    ),
  };
}

function writeLtx2SplitToken(
  output: { cosValues: Float32Array; sinValues: Float32Array },
  tokenShape: {
    batch: number;
    token: number;
    numAttentionHeads: number;
    headDim: number;
    expectedFreqs: number;
    padding: number;
  },
  grid: Ltx2CoordGrid,
  frequencyCount: number,
  theta: number,
): void {
  const flat = new Float32Array(tokenShape.expectedFreqs);
  const flatSin = new Float32Array(tokenShape.expectedFreqs);
  for (let index = 0; index < tokenShape.padding; index += 1) {
    flat[index] = 1;
  }
  for (let frequencyIndex = 0; frequencyIndex < frequencyCount; frequencyIndex += 1) {
    const frequency = theta ** linspaceUnit(frequencyIndex, frequencyCount) * (Math.PI / 2);
    for (let axis = 0; axis < grid.positionDims; axis += 1) {
      const coord =
        grid.values[
          (tokenShape.batch * grid.tokens + tokenShape.token) * grid.positionDims + axis
        ] ?? 0;
      const angle = frequency * (coord * 2 - 1);
      const offset = tokenShape.padding + frequencyIndex * grid.positionDims + axis;
      flat[offset] = Math.cos(angle);
      flatSin[offset] = Math.sin(angle);
    }
  }
  for (let head = 0; head < tokenShape.numAttentionHeads; head += 1) {
    const sourceOffset = head * tokenShape.headDim;
    const outputOffset =
      ((tokenShape.batch * tokenShape.numAttentionHeads + head) * grid.tokens + tokenShape.token) *
      tokenShape.headDim;
    output.cosValues.set(
      flat.subarray(sourceOffset, sourceOffset + tokenShape.headDim),
      outputOffset,
    );
    output.sinValues.set(
      flatSin.subarray(sourceOffset, sourceOffset + tokenShape.headDim),
      outputOffset,
    );
  }
}
