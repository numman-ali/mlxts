import { describe, expect, test } from "bun:test";
import { zeros } from "@mlxts/core";

import {
  affineFreeLayerNorm,
  assertAttention4d,
  assertImage4d,
  assertSequence3d,
  checkedModule,
} from "./tensor-utils";

describe("Stable Diffusion 3 tensor utilities", () => {
  test("returns concrete shape metadata for valid tensors", () => {
    using sequence = zeros([1, 2, 3]);
    using image = zeros([1, 2, 3, 4]);
    using attention = zeros([1, 2, 3, 4]);

    expect(assertSequence3d(sequence, "sequence")).toEqual({
      batch: 1,
      length: 2,
      channels: 3,
    });
    expect(assertImage4d(image, "image")).toEqual({
      batch: 1,
      height: 2,
      width: 3,
      channels: 4,
    });
    expect(assertAttention4d(attention, "attention")).toEqual({
      batch: 1,
      heads: 2,
      length: 3,
      headDim: 4,
    });
    expect(checkedModule(["module"], 0, "modules")).toBe("module");
  });

  test("rejects malformed shapes before tensor math proceeds", () => {
    using vector = zeros([1, 2]);
    using sequence = zeros([1, 2, 3]);

    expect(() => assertSequence3d(vector, "sequence")).toThrow("rank-3");
    expect(() => assertImage4d(sequence, "image")).toThrow("rank-4");
    expect(() => assertAttention4d(sequence, "attention")).toThrow("rank-4");
    expect(() => checkedModule([], 0, "modules")).toThrow("missing module");
    expect(() => affineFreeLayerNorm(sequence, 4, "norm")).toThrow("last dimension");
  });
});
