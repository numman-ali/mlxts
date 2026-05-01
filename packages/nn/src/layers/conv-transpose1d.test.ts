import { describe, expect, test } from "bun:test";
import { grad, MxArray, mxEval, sum } from "@mlxts/core";
import { ConvTranspose1d } from "./conv-transpose1d";

describe("ConvTranspose1d", () => {
  test("forward computes channel-last transposed convolution", () => {
    using conv = new ConvTranspose1d(3, 1, 1, 2, 0, 1, 1);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([1, 1, 1], [1, 1, 3]);
    initialWeight.free();

    const initialBias = conv.bias;
    if (initialBias === null) {
      throw new Error("expected ConvTranspose1d bias to be initialized");
    }
    conv.bias = MxArray.fromData([1], [1]);
    initialBias.free();

    const x = MxArray.fromData([1, 2, 3], [1, 1, 3]);
    const out = conv.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([1, 2, 1]);
    expect(out.toList()).toEqual([[[7], [1]]]);

    x.free();
    out.free();
  });

  test("parameters expose stride, output padding, and bias choices", () => {
    using conv = new ConvTranspose1d(4, 8, 3, 2, 1, 1, 1, 4);
    const params = conv.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).toContain("bias");
    expect(params.weight).toBe(conv.weight);
    expect(conv.inputChannels).toBe(4);
    expect(conv.outputChannels).toBe(8);
    expect(conv.kernelSize).toBe(3);
    expect(conv.stride).toBe(2);
    expect(conv.padding).toBe(1);
    expect(conv.dilation).toBe(1);
    expect(conv.outputPadding).toBe(1);
    expect(conv.groups).toBe(4);
  });

  test("parameters exclude bias when hasBias=false", () => {
    using conv = new ConvTranspose1d(2, 4, 3, 1, 0, 1, 0, 1, false);
    const params = conv.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).not.toContain("bias");
    expect(conv.bias).toBeNull();
  });

  test("invalid dims and invalid inputs throw", () => {
    expect(() => new ConvTranspose1d(0, 4, 3)).toThrow();
    expect(() => new ConvTranspose1d(4, 0, 3)).toThrow();
    expect(() => new ConvTranspose1d(4, 4, 0)).toThrow();
    expect(() => new ConvTranspose1d(4, 4, 3, 0)).toThrow();
    expect(() => new ConvTranspose1d(4, 4, 3, 1, -1)).toThrow();
    expect(() => new ConvTranspose1d(4, 4, 3, 1, 0, 0)).toThrow();
    expect(() => new ConvTranspose1d(4, 4, 3, 1, 0, 1, 1)).toThrow("outputPadding");
    expect(() => new ConvTranspose1d(3, 4, 3, 1, 0, 1, 0, 2)).toThrow("inputChannels");
    expect(() => new ConvTranspose1d(4, 3, 3, 1, 0, 1, 0, 2)).toThrow("outputChannels");

    using conv = new ConvTranspose1d(2, 2, 2);
    const wrongChannels = MxArray.fromData([1, 2, 3], [1, 3, 1]);
    expect(() => conv.forward(wrongChannels)).toThrow("expected last dimension 2");
    wrongChannels.free();

    const wrongRank = MxArray.fromData([1, 2], [2]);
    expect(() => conv.forward(wrongRank)).toThrow("expected rank-3 input");
    wrongRank.free();

    using wrongWeightConv = new ConvTranspose1d(2, 2, 2);
    const previousWeight = wrongWeightConv.weight;
    wrongWeightConv.weight = MxArray.fromData([1, 2, 3, 4], [2, 2]);
    previousWeight.free();
    const validInput = MxArray.fromData([1, 2, 3, 4], [1, 2, 2]);
    expect(() => wrongWeightConv.forward(validInput)).toThrow("expected weight shape");
    validInput.free();

    using wrongBiasConv = new ConvTranspose1d(1, 2, 2);
    const previousBias = wrongBiasConv.bias;
    if (previousBias === null) {
      throw new Error("expected ConvTranspose1d bias to be initialized");
    }
    wrongBiasConv.bias = MxArray.fromData([1], [1]);
    previousBias.free();
    const biasInput = MxArray.fromData([1, 2], [1, 2, 1]);
    expect(() => wrongBiasConv.forward(biasInput)).toThrow("expected bias shape");
    biasInput.free();
  });

  test("gradient flows through convTranspose1d", () => {
    using conv = new ConvTranspose1d(2, 4, 2, 1, 0, 1, 0, 2, false);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6], [1, 3, 2]);

    const lossFn = (input: MxArray) => sum(conv.forward(input));
    const gradFn = grad(lossFn);
    const g = gradFn(x);
    mxEval(g);

    expect(g.shape).toEqual([1, 3, 2]);

    x.free();
    g.free();
  });
});
