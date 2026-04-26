/**
 * Explicit causal mask helpers for decoder inference.
 * @module
 */

import {
  add,
  arange,
  array,
  asType,
  type DType,
  expandDims,
  greater,
  greaterEqual,
  less,
  type MxArray,
  maximum,
  multiply,
  ones,
  subtract,
  triu,
  where,
} from "@mlxts/core";

export type AttentionMask = MxArray | "causal" | null;

export function canOmitLeftPaddedAttentionMask(
  queryLength: number,
  leftPaddingValues: readonly number[],
): boolean {
  return queryLength === 1 && leftPaddingValues.every((padding) => padding === 0);
}

function validateMaskLength(name: string, value: number, context: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${context}: ${name} must be a non-negative integer, got ${value}.`);
  }
}

function trimmedSlidingTotalKeyLength(
  queryLength: number,
  pastLength: number,
  windowSize: number,
): number {
  const fullLength = pastLength + queryLength;
  if (queryLength === 1) {
    return Math.min(windowSize, fullLength);
  }
  return Math.min(fullLength, windowSize + queryLength - 1);
}

function positionRange(start: number, length: number): MxArray {
  if (length === 1) {
    return array([start], "int32");
  }
  return arange(start, start + length, 1, "int32");
}

/** Build an additive causal attention mask for a query block against total keys. */
export function createCausalMask(
  queryLength: number,
  totalKeyLength: number,
  pastLength: number,
  dtype: DType = "float32",
  windowSize?: number,
): MxArray | null {
  if (queryLength <= 0 || totalKeyLength <= 0) {
    throw new Error(
      `createCausalMask: queryLength and totalKeyLength must be positive, got ${queryLength} and ${totalKeyLength}`,
    );
  }

  if (windowSize !== undefined && (!Number.isInteger(windowSize) || windowSize <= 0)) {
    throw new Error(
      `createCausalMask: windowSize must be a positive integer when present, got ${windowSize}`,
    );
  }

  if (queryLength === 1 && totalKeyLength === pastLength + 1) {
    return null;
  }

  if (windowSize !== undefined) {
    using queryOffsets = positionRange(0, queryLength);
    using queryPositions = add(queryOffsets, pastLength);
    using keyPositions = arange(0, totalKeyLength, 1, "int32");
    using queryGrid = expandDims(queryPositions, 1);
    using keyGrid = expandDims(keyPositions, 0);
    using futureMask = greater(keyGrid, queryGrid);
    using shiftedQueryGrid = subtract(queryGrid, windowSize - 1);
    using minVisibleKey = maximum(shiftedQueryGrid, 0);
    using staleMask = less(keyGrid, minVisibleKey);
    using staleAdditive = where(staleMask, -1e9, 0.0);
    using additiveMaskFloat = where(futureMask, -1e9, staleAdditive);
    using additiveMask = additiveMaskFloat.asType(dtype);
    using maskWithBatchAxis = expandDims(additiveMask, 0);
    return expandDims(maskWithBatchAxis, 0);
  }

  using rawMask = ones([queryLength, totalKeyLength], dtype);
  using upperTriangle = triu(rawMask, pastLength + 1);
  using additiveMaskFloat = multiply(upperTriangle, -1e9);
  using additiveMask = additiveMaskFloat.asType(dtype);
  using maskWithBatchAxis = expandDims(additiveMask, 0);
  return expandDims(maskWithBatchAxis, 0);
}

/** Build the mask for a single forward step given the current cache offset. */
export function createStepCausalMask(
  queryLength: number,
  pastLength: number,
  dtype: DType = "float32",
  windowSize?: number,
  trimmedWindow = false,
): MxArray | null {
  const totalKeyLength =
    windowSize === undefined || !trimmedWindow
      ? pastLength + queryLength
      : trimmedSlidingTotalKeyLength(queryLength, pastLength, windowSize);
  const visiblePastLength = Math.max(0, totalKeyLength - queryLength);
  return createCausalMask(queryLength, totalKeyLength, visiblePastLength, dtype, windowSize);
}

function createBooleanCausalMask(
  queryLength: number,
  totalKeyLength: number,
  pastLength: number,
  windowSize?: number,
): MxArray {
  using queryOffsets = positionRange(0, queryLength);
  using queryPositions = add(queryOffsets, pastLength);
  using keyPositions = arange(0, totalKeyLength, 1, "int32");
  using queryGrid = expandDims(queryPositions, 1);
  using keyGrid = expandDims(keyPositions, 0);

  let visibleMask = greaterEqual(queryGrid, keyGrid);
  try {
    if (windowSize !== undefined) {
      using keyWindowLimit = add(keyGrid, windowSize);
      using localMask = less(queryGrid, keyWindowLimit);
      using visibleMaskInts = asType(visibleMask, "int32");
      using localMaskInts = asType(localMask, "int32");
      using combinedMaskInts = multiply(visibleMaskInts, localMaskInts);
      const combinedMask = asType(combinedMaskInts, "bool");
      visibleMask.free();
      visibleMask = combinedMask;
    }

    using maskWithBatchAxis = expandDims(visibleMask, 0);
    return expandDims(maskWithBatchAxis, 0);
  } catch (error) {
    visibleMask.free();
    throw error;
  }
}

/** Build the fastest available attention mask representation for fused attention. */
export function createFastAttentionMask(
  queryLength: number,
  totalKeyLength: number,
  pastLength: number,
  windowSize?: number,
): AttentionMask {
  if (queryLength <= 0 || totalKeyLength <= 0) {
    throw new Error(
      `createFastAttentionMask: queryLength and totalKeyLength must be positive, got ${queryLength} and ${totalKeyLength}`,
    );
  }
  if (windowSize !== undefined && (!Number.isInteger(windowSize) || windowSize <= 0)) {
    throw new Error(
      `createFastAttentionMask: windowSize must be a positive integer when present, got ${windowSize}`,
    );
  }

  if (queryLength === 1 && totalKeyLength === pastLength + 1) {
    return null;
  }
  if (windowSize === undefined || (pastLength === 0 && queryLength <= windowSize)) {
    return "causal";
  }
  return createBooleanCausalMask(queryLength, totalKeyLength, pastLength, windowSize);
}

/** Build a boolean causal mask for left-padded batched decoder inputs. */
export function createLeftPaddedAttentionMask(
  queryLength: number,
  totalKeyLength: number,
  pastLength: number,
  leftPadding: MxArray,
  windowSize?: number,
): MxArray {
  validateMaskLength("queryLength", queryLength, "createLeftPaddedAttentionMask");
  validateMaskLength("totalKeyLength", totalKeyLength, "createLeftPaddedAttentionMask");
  validateMaskLength("pastLength", pastLength, "createLeftPaddedAttentionMask");
  if (queryLength <= 0 || totalKeyLength <= 0) {
    throw new Error(
      `createLeftPaddedAttentionMask: queryLength and totalKeyLength must be positive, got ${queryLength} and ${totalKeyLength}`,
    );
  }
  if (windowSize !== undefined && (!Number.isInteger(windowSize) || windowSize <= 0)) {
    throw new Error(
      `createLeftPaddedAttentionMask: windowSize must be a positive integer when present, got ${windowSize}`,
    );
  }
  if (leftPadding.shape.length !== 1) {
    throw new Error(
      `createLeftPaddedAttentionMask: leftPadding must be rank-1, got rank ${leftPadding.shape.length}.`,
    );
  }

  using queryOffsets = positionRange(pastLength, queryLength);
  using keyPositions = arange(0, totalKeyLength, 1, "int32");
  using queryBatch = expandDims(queryOffsets, 0);
  using keyBatch = expandDims(keyPositions, 0);
  using leftPaddingBatch = expandDims(leftPadding, 1);
  using queryRows = expandDims(queryBatch, 2);
  using keyColumns = expandDims(keyBatch, 0);
  using leftPaddingRows = expandDims(leftPaddingBatch, 2);
  using causalMask = greaterEqual(queryRows, keyColumns);
  using paddingMask = greaterEqual(keyColumns, leftPaddingRows);
  using causalInts = asType(causalMask, "int32");
  using paddingInts = asType(paddingMask, "int32");
  let combinedInts = multiply(causalInts, paddingInts);

  try {
    if (windowSize !== undefined) {
      using keyWindowLimit = add(keyColumns, windowSize);
      using localMask = less(queryRows, keyWindowLimit);
      using localInts = asType(localMask, "int32");
      const nextCombinedInts = multiply(combinedInts, localInts);
      combinedInts.free();
      combinedInts = nextCombinedInts;
    }

    using combinedMask = asType(combinedInts, "bool");
    return expandDims(combinedMask, 1);
  } finally {
    combinedInts.free();
  }
}

/** Build the fast attention mask for one decoder step from the current cache offset. */
export function createStepAttentionMask(
  queryLength: number,
  pastLength: number,
  windowSize?: number,
  trimmedWindow = false,
): AttentionMask {
  const totalKeyLength =
    windowSize === undefined || !trimmedWindow
      ? pastLength + queryLength
      : trimmedSlidingTotalKeyLength(queryLength, pastLength, windowSize);
  const visiblePastLength = Math.max(0, totalKeyLength - queryLength);
  return createFastAttentionMask(queryLength, totalKeyLength, visiblePastLength, windowSize);
}
