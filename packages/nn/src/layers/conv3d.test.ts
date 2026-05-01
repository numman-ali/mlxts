import { describe, expect, test } from "bun:test";

import { array, grad, MxArray, mxEval, ones, sum } from "@mlxts/core";
import { Conv3d } from "./conv3d";

describe("Conv3d", () => {
  test("forward computes dense convolution for channel-last volumes", () => {
    using conv = new Conv3d(1, 1, [2, 2, 2], 1, 0, 1, 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    initialWeight.free();
    using input = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    using output = conv.forward(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 1, 1, 1, 1]);
    expect(output.toList()).toEqual([[[[[204]]]]]);
  });

  test("forward applies bias", () => {
    using conv = new Conv3d(2, 2, 1);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([2, 3, 5, 7], [2, 1, 1, 1, 2]);
    initialWeight.free();
    if (conv.bias === null) {
      throw new Error("expected Conv3d bias to be initialized");
    }
    const initialBias = conv.bias;
    conv.bias = array([10, 20]);
    initialBias.free();
    using input = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 2, 2]);
    using output = conv.forward(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 1, 1, 2, 2]);
    expect(output.toList()).toEqual([
      [
        [
          [
            [18, 39],
            [28, 63],
          ],
        ],
      ],
    ]);
  });

  test("parameters include weight and optional bias", () => {
    using conv = new Conv3d(3, 4, 3);
    expect(conv.parameters()).toHaveProperty("weight");
    expect(conv.parameters()).toHaveProperty("bias");

    using withoutBias = new Conv3d(3, 4, 3, 1, 1, 1, 1, false);
    expect(withoutBias.parameters()).toHaveProperty("weight");
    expect(withoutBias.parameters()).not.toHaveProperty("bias");
  });

  test("exposes normalized configuration", () => {
    using conv = new Conv3d(3, 4, [1, 2, 3], [2, 1, 1], [0, 1, 2], [1, 2, 1]);

    expect(conv.inputChannels).toBe(3);
    expect(conv.outputChannels).toBe(4);
    expect(conv.kernelSize).toEqual([1, 2, 3]);
    expect(conv.stride).toEqual([2, 1, 1]);
    expect(conv.padding).toEqual([0, 1, 2]);
    expect(conv.dilation).toEqual([1, 2, 1]);
    expect(conv.groups).toBe(1);
  });

  test("invalid constructor and forward shapes throw", () => {
    expect(() => new Conv3d(0, 4, 3)).toThrow("inputChannels");
    expect(() => new Conv3d(4, 0, 3)).toThrow("outputChannels");
    expect(() => new Conv3d(4, 4, 0)).toThrow("kernelSize");
    expect(() => new Conv3d(4, 4, 3, 0)).toThrow("stride");
    expect(() => new Conv3d(4, 4, 3, 1, -1)).toThrow("padding");
    expect(() => new Conv3d(4, 4, 3, 1, 0, 0)).toThrow("dilation");
    expect(() => new Conv3d(4, 4, 3, 1, 0, 1, 2)).toThrow("groups other than 1");
    using conv = new Conv3d(2, 2, 2);
    using wrongInput = ones([1, 2, 2, 2, 3]);
    expect(() => conv.forward(wrongInput)).toThrow("last dimension");
  });

  test("validates weight and bias replacement shapes", () => {
    using conv = new Conv3d(1, 2, 2);
    const initialWeight = conv.weight;
    conv.weight = ones([2, 1, 1, 1, 1]);
    initialWeight.free();
    using input = ones([1, 2, 2, 2, 1]);
    expect(() => conv.forward(input)).toThrow("weight shape");

    const wrongWeight = conv.weight;
    conv.weight = ones([2, 2, 2, 2, 1]);
    wrongWeight.free();
    const initialBias = conv.bias;
    conv.bias = ones([3]);
    initialBias?.free();
    expect(() => conv.forward(input)).toThrow("bias shape");
  });

  test("gradient flows through conv3d", () => {
    using conv = new Conv3d(1, 2, 2, 1, 0, 1, 1, false);
    using input = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);

    const lossFn = (x: MxArray) => sum(conv.forward(x));
    const gradFn = grad(lossFn);
    using gradient = gradFn(input);
    mxEval(gradient);

    expect(gradient.shape).toEqual([1, 2, 2, 2, 1]);
  });
});
