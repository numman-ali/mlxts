/**
 * Gradient-tree utilities shared across training loops.
 *
 * @module
 */

import type { MxArray, ParameterTree } from "@mlxts/core";
import {
  add,
  multiply,
  mxEval,
  square,
  sum,
  treeFlatten,
  treeLeaves,
  treeUnflatten,
} from "@mlxts/core";

type FlatGradientEntry = [path: string[], value: MxArray];

function pathKey(path: readonly string[]): string {
  return path.join(".");
}

function formatGradientPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

function assertMatchingGradientEntries(
  left: readonly FlatGradientEntry[],
  right: readonly FlatGradientEntry[],
  context: string,
): void {
  if (left.length !== right.length) {
    throw new Error(`${context}: gradient tree leaf counts do not match`);
  }

  for (let index = 0; index < left.length; index++) {
    const leftEntry = left[index];
    const rightEntry = right[index];
    if (leftEntry === undefined || rightEntry === undefined) {
      throw new Error(`${context}: missing gradient leaf at index ${index}`);
    }
    if (pathKey(leftEntry[0]) !== pathKey(rightEntry[0])) {
      throw new Error(
        `${context}: gradient path mismatch at index ${index} (${formatGradientPath(leftEntry[0])} vs ${formatGradientPath(rightEntry[0])})`,
      );
    }
  }
}

function mapGradientEntries(
  entries: readonly FlatGradientEntry[],
  mapper: (value: MxArray, path: readonly string[]) => MxArray,
  context: string,
): FlatGradientEntry[] {
  const mapped: FlatGradientEntry[] = [];
  try {
    for (const [path, value] of entries) {
      mapped.push([[...path], mapper(value, path)]);
    }
    return mapped;
  } catch (error) {
    for (const [, value] of mapped) {
      value.free();
    }
    throw new Error(
      error instanceof Error ? `${context}: ${error.message}` : `${context}: ${String(error)}`,
    );
  }
}

function gradientEntriesToTree(entries: FlatGradientEntry[], context: string): ParameterTree {
  try {
    return treeUnflatten(entries);
  } catch (error) {
    for (const [, value] of entries) {
      value.free();
    }
    throw new Error(
      error instanceof Error ? `${context}: ${error.message}` : `${context}: ${String(error)}`,
    );
  }
}

/** Free every gradient leaf in a tree. */
export function freeGradientTree(tree: ParameterTree): void {
  for (const [, value] of treeFlatten(tree)) {
    value.free();
  }
}

/** Force evaluation of all gradient leaves in a tree. */
export function evalGradientTree(tree: ParameterTree): void {
  const leaves = treeLeaves(tree);
  if (leaves.length > 0) {
    mxEval(...leaves);
  }
}

/** Add two gradient trees with matching structure. */
export function accumulateGradients(
  accumulated: ParameterTree,
  next: ParameterTree,
): ParameterTree {
  const accumulatedEntries = treeFlatten(accumulated);
  const nextEntries = treeFlatten(next);
  assertMatchingGradientEntries(accumulatedEntries, nextEntries, "train.accumulateGradients");

  const summedEntries: FlatGradientEntry[] = [];
  try {
    for (let index = 0; index < accumulatedEntries.length; index++) {
      const accumulatedEntry = accumulatedEntries[index];
      const nextEntry = nextEntries[index];
      if (accumulatedEntry === undefined || nextEntry === undefined) {
        throw new Error(`missing gradient leaf at index ${index}`);
      }
      summedEntries.push([[...accumulatedEntry[0]], add(accumulatedEntry[1], nextEntry[1])]);
    }
  } catch (error) {
    for (const [, value] of summedEntries) {
      value.free();
    }
    throw new Error(
      error instanceof Error
        ? `train.accumulateGradients: ${error.message}`
        : `train.accumulateGradients: ${String(error)}`,
    );
  }

  return gradientEntriesToTree(summedEntries, "train.accumulateGradients");
}

export const accumulateGradientTrees = accumulateGradients;

/** Multiply every gradient leaf by a scalar factor. */
export function scaleGradientTree(tree: ParameterTree, factor: number): ParameterTree {
  if (factor === 1) {
    return tree;
  }

  const entries = treeFlatten(tree);
  const scaledEntries = mapGradientEntries(
    entries,
    (grad) => multiply(grad, factor),
    "train.scaleGradientTree",
  );
  return gradientEntriesToTree(scaledEntries, "train.scaleGradientTree");
}

/** Euclidean norm over every gradient leaf. */
export function gradientNorm(tree: ParameterTree): number {
  let totalSquared = 0;
  for (const grad of treeLeaves(tree)) {
    using squared = square(grad);
    using squaredSum = sum(squared);
    mxEval(squaredSum);
    totalSquared += squaredSum.item();
  }
  return Math.sqrt(totalSquared);
}

/** Clip a gradient tree by global norm when it exceeds a threshold. */
export function clipGradientTree(tree: ParameterTree, maxGradNorm: number | null): ParameterTree {
  if (maxGradNorm === null) {
    return tree;
  }

  const norm = gradientNorm(tree);
  if (!Number.isFinite(norm)) {
    throw new Error(`train: gradient norm is non-finite (${String(norm)})`);
  }
  if (norm === 0 || norm <= maxGradNorm) {
    return tree;
  }

  return scaleGradientTree(tree, maxGradNorm / norm);
}
