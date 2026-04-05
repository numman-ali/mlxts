import { describe, expect, test } from "bun:test";

import { full, type ParameterTree } from "@mlxts/core";

import { WeightMismatchError } from "../types";
import {
  assignWeightPath,
  describeWeightShape,
  formatParameterPaths,
  listParameterPaths,
} from "./weight-assignment";

function freeTree(tree: ParameterTree): void {
  for (const value of Object.values(tree)) {
    if (value instanceof Object && "free" in value && typeof value.free === "function") {
      value.free();
    } else if (typeof value === "object" && value !== null) {
      freeTree(value as ParameterTree);
    }
  }
}

describe("weight assignment helpers", () => {
  test("assignWeightPath replaces nested parameters and disposes the previous tensor", () => {
    const weight = full([2], 1);
    const bias = full([1], 2);
    const layerWeight = full([1], 3);
    const replacement = full([2], 9);
    const layerReplacement = full([1], 7);
    const model = {
      weight,
      nested: { bias },
      layers: [{ weight: layerWeight }],
    };

    try {
      assignWeightPath(model, "weight", replacement);
      assignWeightPath(model, "layers.0.weight", layerReplacement);

      expect(weight.isDisposed).toBe(true);
      expect(layerWeight.isDisposed).toBe(true);
      expect(model.weight).toBe(replacement);
      expect(model.layers[0]?.weight).toBe(layerReplacement);
    } finally {
      replacement.free();
      layerReplacement.free();
      bias.free();
    }
  });

  test("assignWeightPath rejects malformed paths and mismatched shapes", () => {
    const model = {
      weight: full([2], 1),
      nested: { bias: full([1], 2) },
      layers: [{ weight: full([1], 3) }],
    };
    const mismatch = full([3], 1);
    const replacement = full([1], 4);

    try {
      expect(() => assignWeightPath(model, "", replacement)).toThrow("path must not be empty");
      expect(() => assignWeightPath(model, "layers.bad.weight", replacement)).toThrow(
        'segment "bad" is not a valid array index',
      );
      expect(() => assignWeightPath(model, "weight.value", replacement)).toThrow(
        "does not point to an MxArray parameter",
      );
      expect(() => assignWeightPath(model, "nested", replacement)).toThrow(
        "does not point to an MxArray parameter",
      );
      expect(() => assignWeightPath(model, "weight", mismatch)).toThrow(WeightMismatchError);
    } finally {
      mismatch.free();
      replacement.free();
      model.weight.free();
      model.nested.bias.free();
      model.layers[0]?.weight.free();
    }
  });

  test("format and enumerate parameter paths", () => {
    const weight = full([2], 1);
    const tree: ParameterTree = {
      weight,
      nested: {
        bias: full([1], 2),
      },
    };

    try {
      expect(listParameterPaths(tree)).toEqual(["weight", "nested.bias"]);
      expect(formatParameterPaths(["nested.bias", "weight"])).toBe("nested.bias, weight");
      expect(describeWeightShape(weight)).toBe("[2]");
    } finally {
      freeTree(tree);
    }
  });
});
