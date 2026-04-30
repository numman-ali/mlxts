import { describe, expect, test } from "bun:test";

import { array, MxArray, mxEval, ones } from "@mlxts/core";
import { Conv2d } from "./conv2d";

describe("Conv2d", () => {
  test("forward computes dense convolution for channel-last images", () => {
    using conv = new Conv2d(1, 1, [2, 2], 1, 0, 1, 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    initialWeight.free();
    using input = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using output = conv.forward(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 1, 1, 1]);
    expect(output.toList()).toEqual([[[[30]]]]);
  });

  test("forward applies bias and grouped filters", () => {
    using conv = new Conv2d(2, 2, 1, 1, 0, 1, 2);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([2, 3], [2, 1, 1, 1]);
    initialWeight.free();
    if (conv.bias === null) {
      throw new Error("expected Conv2d bias to be initialized");
    }
    const initialBias = conv.bias;
    conv.bias = array([10, 20]);
    initialBias.free();
    using input = MxArray.fromData([1, 2, 3, 4], [1, 1, 2, 2]);
    using output = conv.forward(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 1, 2, 2]);
    expect(output.toList()).toEqual([
      [
        [
          [12, 26],
          [16, 32],
        ],
      ],
    ]);
  });

  test("parameters include weight and optional bias", () => {
    using conv = new Conv2d(3, 4, 3);
    expect(conv.parameters()).toHaveProperty("weight");
    expect(conv.parameters()).toHaveProperty("bias");

    using withoutBias = new Conv2d(3, 4, 3, 1, 1, 1, 1, false);
    expect(withoutBias.parameters()).toHaveProperty("weight");
    expect(withoutBias.parameters()).not.toHaveProperty("bias");
  });

  test("invalid constructor and forward shapes throw", () => {
    expect(() => new Conv2d(0, 4, 3)).toThrow("inputChannels");
    expect(() => new Conv2d(4, 0, 3)).toThrow("outputChannels");
    expect(() => new Conv2d(4, 4, 0)).toThrow("kernelSize");
    expect(() => new Conv2d(4, 4, 3, 0)).toThrow("stride");
    expect(() => new Conv2d(4, 4, 3, 1, -1)).toThrow("padding");
    expect(() => new Conv2d(4, 4, 3, 1, 0, 0)).toThrow("dilation");
    expect(() => new Conv2d(3, 4, 3, 1, 0, 1, 2)).toThrow("groups");
    using conv = new Conv2d(2, 2, 2);
    using wrongInput = ones([1, 2, 2, 3]);
    expect(() => conv.forward(wrongInput)).toThrow("last dimension");
  });

  test("validates weight and bias replacement shapes", () => {
    using conv = new Conv2d(1, 2, 2);
    const initialWeight = conv.weight;
    conv.weight = ones([2, 1, 1, 1]);
    initialWeight.free();
    using input = ones([1, 2, 2, 1]);
    expect(() => conv.forward(input)).toThrow("weight shape");

    const wrongWeight = conv.weight;
    conv.weight = ones([2, 2, 2, 1]);
    wrongWeight.free();
    const initialBias = conv.bias;
    conv.bias = ones([3]);
    initialBias?.free();
    expect(() => conv.forward(input)).toThrow("bias shape");
  });
});
