/**
 * Qwen 3.5 text rotary embedding helpers.
 * @module
 */

import {
  add,
  array,
  asType,
  broadcastTo,
  concatenate,
  cos,
  type DType,
  expandDims,
  formatShape,
  type MxArray,
  multiply,
  negative,
  reshape,
  retainArray,
  sin,
  slice,
  where,
} from "@mlxts/core";

import type { Qwen3_5TextConfig } from "./types";

function takeLastAxisRange(x: MxArray, start: number, end: number): MxArray {
  const rank = x.shape.length;
  const batch = x.shape[0];
  const sequenceLength = x.shape[rank - 2];
  const lastDimension = x.shape[rank - 1];
  if (
    batch === undefined ||
    sequenceLength === undefined ||
    lastDimension === undefined ||
    rank < 3
  ) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding: expected rank >= 3 tensor, got ${formatShape(x.shape)}.`,
    );
  }

  const startIndices = Array(rank).fill(0);
  const stopIndices = [...x.shape];
  startIndices[rank - 1] = start;
  stopIndices[rank - 1] = end;
  return slice(x, startIndices, stopIndices);
}

function takeAxis0Slice(x: MxArray, axisIndex: number): MxArray {
  const [axisCount, batch, sequenceLength] = x.shape;
  if (
    axisCount === undefined ||
    batch === undefined ||
    sequenceLength === undefined ||
    x.shape.length !== 3
  ) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding: expected position ids with shape [axis, batch, seq], got ${formatShape(x.shape)}.`,
    );
  }
  if (axisIndex < 0 || axisIndex >= axisCount) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding: axis ${axisIndex} is out of range for ${axisCount} position-id axes.`,
    );
  }
  using view = slice(x, [axisIndex, 0, 0], [axisIndex + 1, batch, sequenceLength]);
  return reshape(view, [batch, sequenceLength]);
}

function createAxisReplacementMask(
  rotaryDimensions: number,
  axisIndex: 1 | 2,
  sectionSize: number,
): MxArray {
  const rotaryPairs = rotaryDimensions / 2;
  const replacementLimit = sectionSize * 3;
  const mask: number[] = Array.from({ length: rotaryPairs }, (_, pairIndex) =>
    pairIndex < replacementLimit && pairIndex % 3 === axisIndex ? 1 : 0,
  );
  return array(mask, "bool");
}

function rotateHalf(x: MxArray): MxArray {
  const rotaryDimensions = x.shape[x.shape.length - 1];
  if (rotaryDimensions === undefined || rotaryDimensions % 2 !== 0) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding.rotateHalf: expected an even last dimension, got ${rotaryDimensions ?? "undefined"} for shape ${formatShape(x.shape)}.`,
    );
  }

  const half = rotaryDimensions / 2;
  using left = takeLastAxisRange(x, 0, half);
  using right = takeLastAxisRange(x, half, rotaryDimensions);
  using negRight = negative(right);
  return concatenate([negRight, left], x.shape.length - 1);
}

function applyRotaryToTensor(
  x: MxArray,
  cosEmbeddings: MxArray,
  sinEmbeddings: MxArray,
  rotaryDimensions: number,
): MxArray {
  const lastDimension = x.shape[x.shape.length - 1];
  const sequenceLength = x.shape[x.shape.length - 2];
  const batchSize = x.shape[0];
  if (
    lastDimension === undefined ||
    sequenceLength === undefined ||
    batchSize === undefined ||
    x.shape.length !== 4
  ) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding.applyRotaryToTensor: expected [batch, heads, seq, dim], got ${formatShape(x.shape)}.`,
    );
  }
  if (rotaryDimensions > lastDimension) {
    throw new Error(
      `Qwen3_5TextRotaryEmbedding.applyRotaryToTensor: rotary dimensions ${rotaryDimensions} exceed last dimension ${lastDimension}.`,
    );
  }

  using rotaryInputs = takeLastAxisRange(x, 0, rotaryDimensions);
  using cosHeads = expandDims(cosEmbeddings, 1);
  using sinHeads = expandDims(sinEmbeddings, 1);
  using rotatedHalf = rotateHalf(rotaryInputs);
  using scaledInputs = multiply(rotaryInputs, cosHeads);
  using scaledRotated = multiply(rotatedHalf, sinHeads);
  using rotated = add(scaledInputs, scaledRotated);
  if (rotaryDimensions === lastDimension) {
    return retainArray(rotated);
  }

  using passThrough = takeLastAxisRange(x, rotaryDimensions, lastDimension);
  return concatenate([rotated, passThrough], 3);
}

type RotaryEmbeddings = {
  cos: MxArray;
  sin: MxArray;
};

type AppliedRotaryEmbeddings = {
  queries: MxArray;
  keys: MxArray;
};

/** Multimodal-aware rotary helper for Qwen 3.5 text attention. */
export class Qwen3_5TextRotaryEmbedding {
  #invFreq: MxArray;
  #rotaryDimensions: number;
  #heightMask: MxArray | null;
  #widthMask: MxArray | null;

  constructor(config: Qwen3_5TextConfig) {
    const rotaryDimensions = Math.floor(config.headDim * config.partialRotaryFactor);
    if (
      !Number.isInteger(rotaryDimensions) ||
      rotaryDimensions <= 0 ||
      rotaryDimensions % 2 !== 0
    ) {
      throw new Error(
        `Qwen3_5TextRotaryEmbedding: partial rotary dimensions must be a positive even integer, got ${rotaryDimensions}.`,
      );
    }

    const values = Array.from(
      { length: rotaryDimensions / 2 },
      (_, index) => config.ropeParameters.ropeTheta ** (-(2 * index) / rotaryDimensions),
    );
    this.#invFreq = array(values, "float32");
    this.#rotaryDimensions = rotaryDimensions;
    this.#heightMask =
      (config.ropeParameters.mropeSection[1] ?? 0) > 0
        ? createAxisReplacementMask(rotaryDimensions, 1, config.ropeParameters.mropeSection[1] ?? 0)
        : null;
    this.#widthMask =
      (config.ropeParameters.mropeSection[2] ?? 0) > 0
        ? createAxisReplacementMask(rotaryDimensions, 2, config.ropeParameters.mropeSection[2] ?? 0)
        : null;
  }

  get rotaryDimensions(): number {
    return this.#rotaryDimensions;
  }

  embeddings(x: MxArray, positionIds: MxArray): RotaryEmbeddings {
    const expandedPositionIds = this.expandPositionIds(positionIds);
    try {
      return this.buildEmbeddings(x.dtype, expandedPositionIds);
    } finally {
      expandedPositionIds.free();
    }
  }

  apply(queryStates: MxArray, keyStates: MxArray, positionIds: MxArray): AppliedRotaryEmbeddings {
    const embeddings = this.embeddings(queryStates, positionIds);
    try {
      const queries = applyRotaryToTensor(
        queryStates,
        embeddings.cos,
        embeddings.sin,
        this.#rotaryDimensions,
      );
      const keys = applyRotaryToTensor(
        keyStates,
        embeddings.cos,
        embeddings.sin,
        this.#rotaryDimensions,
      );
      return {
        queries,
        keys,
      };
    } finally {
      embeddings.cos.free();
      embeddings.sin.free();
    }
  }

  [Symbol.dispose](): void {
    this.#invFreq.free();
    this.#heightMask?.free();
    this.#widthMask?.free();
  }

  private expandPositionIds(positionIds: MxArray): MxArray {
    const [batchSize, sequenceLength] = positionIds.shape;
    if (positionIds.shape.length === 2 && batchSize !== undefined && sequenceLength !== undefined) {
      using base = reshape(positionIds, [1, batchSize, sequenceLength]);
      return broadcastTo(base, [3, batchSize, sequenceLength]);
    }

    const [axes, expandedBatchSize, expandedSequenceLength] = positionIds.shape;
    if (
      positionIds.shape.length !== 3 ||
      axes !== 3 ||
      expandedBatchSize === undefined ||
      expandedSequenceLength === undefined
    ) {
      throw new Error(
        `Qwen3_5TextRotaryEmbedding.embeddings: expected position ids with shape [batch, seq] or [3, batch, seq], got ${formatShape(positionIds.shape)}.`,
      );
    }

    return retainArray(positionIds);
  }

  private buildEmbeddings(dtype: DType, positionIds: MxArray): RotaryEmbeddings {
    const [, batchSize, sequenceLength] = positionIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error(
        `Qwen3_5TextRotaryEmbedding.buildEmbeddings: position ids are missing batch or sequence dimensions: ${formatShape(positionIds.shape)}.`,
      );
    }

    let mergedFrequencies: MxArray | null = null;
    try {
      using timePositions = takeAxis0Slice(positionIds, 0);
      using timeFrequencies = this.axisFrequencies(timePositions, batchSize, sequenceLength);
      mergedFrequencies = this.applyAxisOverrides(
        timeFrequencies,
        positionIds,
        batchSize,
        sequenceLength,
      );

      using doubled = concatenate([mergedFrequencies, mergedFrequencies], 2);
      using cosValues = cos(doubled);
      using sinValues = sin(doubled);
      return {
        cos: asType(cosValues, dtype),
        sin: asType(sinValues, dtype),
      };
    } finally {
      mergedFrequencies?.free();
    }
  }

  private axisFrequencies(positions: MxArray, batchSize: number, sequenceLength: number): MxArray {
    using positionsView = reshape(positions, [batchSize, sequenceLength, 1]);
    using frequenciesView = reshape(this.#invFreq, [1, 1, this.#rotaryDimensions / 2]);
    return multiply(positionsView, frequenciesView);
  }

  private applyAxisOverrides(
    baseFrequencies: MxArray,
    positionIds: MxArray,
    batchSize: number,
    sequenceLength: number,
  ): MxArray {
    let mergedFrequencies = retainArray(baseFrequencies);
    try {
      for (const [axisIndex, axisMask] of [
        [1, this.#heightMask],
        [2, this.#widthMask],
      ] as const) {
        if (axisMask === null) {
          continue;
        }

        using axisPositions = takeAxis0Slice(positionIds, axisIndex);
        using axisFrequencies = this.axisFrequencies(axisPositions, batchSize, sequenceLength);
        using maskView = reshape(axisMask, [1, 1, this.#rotaryDimensions / 2]);
        using mask = broadcastTo(maskView, [batchSize, sequenceLength, this.#rotaryDimensions / 2]);
        const updated = where(mask, axisFrequencies, mergedFrequencies);
        mergedFrequencies.free();
        mergedFrequencies = updated;
      }
      return retainArray(mergedFrequencies);
    } finally {
      mergedFrequencies.free();
    }
  }
}

/** Build standard sequential Qwen 3.5 position ids with the same value across all 3 axes. */
export function createShiftedQwen3_5PositionIds(
  startPositions: readonly number[],
  sequenceLength: number,
): MxArray {
  const batchSize = startPositions.length;
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error(
      `createSequentialQwen3_5PositionIds: batchSize must be a positive integer, got ${batchSize}.`,
    );
  }
  if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) {
    throw new Error(
      `createSequentialQwen3_5PositionIds: sequenceLength must be a positive integer, got ${sequenceLength}.`,
    );
  }

  const positions = startPositions.map((startPosition, batchIndex) => {
    if (!Number.isInteger(startPosition)) {
      throw new Error(
        `createSequentialQwen3_5PositionIds: startPositions[${batchIndex}] must be an integer, got ${startPosition}.`,
      );
    }
    return Array.from({ length: sequenceLength }, (_, index) => startPosition + index);
  });
  using positionArray = array(positions, "int32");
  using positionView = reshape(positionArray, [1, batchSize, sequenceLength]);
  return broadcastTo(positionView, [3, batchSize, sequenceLength]);
}

/** Build standard sequential Qwen 3.5 position ids for a shared start position. */
export function createSequentialQwen3_5PositionIds(
  batchSize: number,
  sequenceLength: number,
  startPosition: number,
): MxArray {
  return createShiftedQwen3_5PositionIds(Array(batchSize).fill(startPosition), sequenceLength);
}
