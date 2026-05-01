import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, zeros } from "@mlxts/core";

import { LtxVideoCausalConv3d } from "./autoencoder-volume";

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

function setUnitTemporalKernel(conv: LtxVideoCausalConv3d): void {
  conv.conv.weight.free();
  conv.conv.weight = MxArray.fromData([1, 1, 1], [1, 3, 1, 1, 1]);
  conv.conv.bias?.free();
  conv.conv.bias = zeros([1]);
}

describe("LtxVideoCausalConv3d", () => {
  test("pads causal temporal input by repeating the first frame", () => {
    using conv = new LtxVideoCausalConv3d(1, 1, [3, 1, 1], 1, 1, true);
    setUnitTemporalKernel(conv);
    using video = MxArray.fromData([1, 2, 4], [1, 3, 1, 1, 1]);
    using output = conv.forward(video);

    mxEval(output);
    expect(output.shape).toEqual([1, 3, 1, 1, 1]);
    expectCloseList(output.toTypedArray(), [3, 4, 7]);
  });

  test("pads non-causal temporal input by repeating both edges", () => {
    using conv = new LtxVideoCausalConv3d(1, 1, [3, 1, 1], 1, 1, false);
    setUnitTemporalKernel(conv);
    using video = MxArray.fromData([1, 2, 4], [1, 3, 1, 1, 1]);
    using output = conv.forward(video);

    mxEval(output);
    expect(output.shape).toEqual([1, 3, 1, 1, 1]);
    expectCloseList(output.toTypedArray(), [4, 7, 10]);
  });
});
