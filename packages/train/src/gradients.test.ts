import { describe, expect, test } from "bun:test";

import { full, type ParameterTree, treeFlatten } from "@mlxts/core";

import {
  accumulateGradients,
  clipGradientTree,
  gradientNorm,
  scaleGradientTree,
} from "./gradients";

function leafScalar(tree: ParameterTree, path: string): number {
  const entry = treeFlatten(tree).find(([currentPath]) => currentPath.join(".") === path);
  if (entry === undefined) {
    throw new Error(`missing gradient leaf "${path}"`);
  }
  return entry[1].item();
}

function freeTree(tree: ParameterTree): void {
  for (const [, value] of treeFlatten(tree)) {
    value.free();
  }
}

describe("gradients", () => {
  test("accumulateGradients adds matching gradient trees", () => {
    const left: ParameterTree = { weight: full([1], 2), bias: full([1], 3) };
    const right: ParameterTree = { weight: full([1], 5), bias: full([1], 7) };

    try {
      const summed = accumulateGradients(left, right);
      try {
        expect(leafScalar(summed, "weight")).toBe(7);
        expect(leafScalar(summed, "bias")).toBe(10);
      } finally {
        freeTree(summed);
      }
    } finally {
      freeTree(left);
      freeTree(right);
    }
  });

  test("scaleGradientTree scales every leaf by the provided factor", () => {
    const tree: ParameterTree = { weight: full([1], 8) };

    try {
      const scaled = scaleGradientTree(tree, 0.25);
      try {
        expect(leafScalar(scaled, "weight")).toBe(2);
      } finally {
        freeTree(scaled);
      }
    } finally {
      freeTree(tree);
    }
  });

  test("clipGradientTree clips by global norm", () => {
    const tree: ParameterTree = { weight: full([1], 10) };

    try {
      expect(gradientNorm(tree)).toBe(10);
      const clipped = clipGradientTree(tree, 5);
      try {
        expect(leafScalar(clipped, "weight")).toBeCloseTo(5);
      } finally {
        freeTree(clipped);
      }
    } finally {
      freeTree(tree);
    }
  });
});
