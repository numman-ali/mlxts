import { describe, expect, test } from "bun:test";
import { grad, MxArray, mxEval, sum } from "@mlxts/core";
import { Conv1d } from "./conv1d";

describe("Conv1d", () => {
  test("forward computes dense convolution for channel-last inputs", () => {
    using conv = new Conv1d(1, 1, 2);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 2], [1, 2, 1]);
    initialWeight.free();

    const initialBias = conv.bias;
    if (initialBias === null) {
      throw new Error("expected Conv1d bias to be initialized");
    }
    conv.bias = MxArray.fromData([1], [1]);
    initialBias.free();

    const x = MxArray.fromData([1, 2, 3, 4], [1, 4, 1]);
    const out = conv.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([1, 3, 1]);
    expect(out.toList()).toEqual([[[6], [9], [12]]]);

    x.free();
    out.free();
  });

  test("forward applies padding and dilation", () => {
    using conv = new Conv1d(1, 1, 2, 1, 1, 2, 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 10], [1, 2, 1]);
    initialWeight.free();

    const x = MxArray.fromData([1, 2, 3, 4], [1, 4, 1]);
    const out = conv.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([1, 4, 1]);
    expect(out.toList()).toEqual([[[20], [31], [42], [3]]]);

    x.free();
    out.free();
  });

  test("forward supports depthwise groups with channel multipliers", () => {
    using conv = new Conv1d(2, 4, 2, 1, 0, 1, 2, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 0, 0, 1, 1, 1, -1, 1], [4, 2, 1]);
    initialWeight.free();

    const x = MxArray.fromData([1, 10, 2, 20, 3, 30], [1, 3, 2]);
    const out = conv.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([1, 2, 4]);
    expect(out.toList()).toEqual([
      [
        [1, 2, 30, 10],
        [2, 3, 50, 10],
      ],
    ]);

    x.free();
    out.free();
  });

  test("parameters include weight and bias", () => {
    using conv = new Conv1d(4, 8, 3, 2, 1, 2, 4);
    const params = conv.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).toContain("bias");
    expect(params.weight).toBe(conv.weight);
    expect(conv.inputChannels).toBe(4);
    expect(conv.outputChannels).toBe(8);
    expect(conv.kernelSize).toBe(3);
    expect(conv.stride).toBe(2);
    expect(conv.padding).toBe(1);
    expect(conv.dilation).toBe(2);
    expect(conv.groups).toBe(4);
  });

  test("parameters exclude bias when hasBias=false", () => {
    using conv = new Conv1d(2, 4, 3, 1, 0, 1, 1, false);
    const params = conv.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).not.toContain("bias");
    expect(conv.bias).toBeNull();
  });

  test("invalid dims and invalid inputs throw", () => {
    expect(() => new Conv1d(0, 4, 3)).toThrow();
    expect(() => new Conv1d(4, 0, 3)).toThrow();
    expect(() => new Conv1d(4, 4, 0)).toThrow();
    expect(() => new Conv1d(4, 4, 3, 0)).toThrow();
    expect(() => new Conv1d(4, 4, 3, 1, -1)).toThrow();
    expect(() => new Conv1d(4, 4, 3, 1, 0, 0)).toThrow();
    expect(() => new Conv1d(3, 4, 3, 1, 0, 1, 2)).toThrow("inputChannels");
    expect(() => new Conv1d(4, 3, 3, 1, 0, 1, 2)).toThrow("outputChannels");

    using conv = new Conv1d(2, 2, 2);
    const wrongChannels = MxArray.fromData([1, 2, 3], [1, 3, 1]);
    expect(() => conv.forward(wrongChannels)).toThrow("expected last dimension 2");
    wrongChannels.free();

    const wrongRank = MxArray.fromData([1, 2], [2]);
    expect(() => conv.forward(wrongRank)).toThrow("expected rank-3 input");
    wrongRank.free();

    using wrongWeightConv = new Conv1d(2, 2, 2);
    const previousWeight = wrongWeightConv.weight;
    wrongWeightConv.weight = MxArray.fromData([1, 2, 3, 4], [2, 2]);
    previousWeight.free();
    const validInput = MxArray.fromData([1, 2, 3, 4], [1, 2, 2]);
    expect(() => wrongWeightConv.forward(validInput)).toThrow("expected weight shape");
    validInput.free();

    using wrongBiasConv = new Conv1d(1, 2, 2);
    const previousBias = wrongBiasConv.bias;
    if (previousBias === null) {
      throw new Error("expected Conv1d bias to be initialized");
    }
    wrongBiasConv.bias = MxArray.fromData([1], [1]);
    previousBias.free();
    const biasInput = MxArray.fromData([1, 2], [1, 2, 1]);
    expect(() => wrongBiasConv.forward(biasInput)).toThrow("expected bias shape");
    biasInput.free();
  });

  test("gradient flows through conv1d", () => {
    using conv = new Conv1d(2, 4, 2, 1, 0, 1, 2, false);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6], [1, 3, 2]);

    const lossFn = (input: MxArray) => sum(conv.forward(input));
    const gradFn = grad(lossFn);
    const g = gradFn(x);
    mxEval(g);

    expect(g.shape).toEqual([1, 3, 2]);

    x.free();
    g.free();
  });

  test("uses the latest weight parameter after replacement", () => {
    using conv = new Conv1d(1, 1, 2, 1, 0, 1, 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 0], [1, 2, 1]);
    initialWeight.free();

    const x = MxArray.fromData([1, 2, 3], [1, 3, 1]);
    const firstOut = conv.forward(x);
    mxEval(firstOut);
    expect(firstOut.toList()).toEqual([[[1], [2]]]);
    firstOut.free();

    const previousWeight = conv.weight;
    conv.weight = MxArray.fromData([0, 1], [1, 2, 1]);
    previousWeight.free();

    const secondOut = conv.forward(x);
    mxEval(secondOut);
    expect(secondOut.toList()).toEqual([[[2], [3]]]);

    secondOut.free();
    x.free();
  });
});
