import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, ones } from "@mlxts/core";

import {
  QwenImageCausalConv3d,
  QwenImageResample,
  qwenImageNcfhwToNdhwc,
  qwenImageNdhwcToNcfhw,
} from "./autoencoder-blocks";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Qwen-Image VAE blocks", () => {
  test("converts public NCFHW volumes to internal NDHWC and back", () => {
    using channelFirst = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 1, 2, 2]);
    using channelLast = qwenImageNcfhwToNdhwc(channelFirst);
    using roundTrip = qwenImageNdhwcToNcfhw(channelLast);

    mxEval(channelLast, roundTrip);
    expect(channelLast.shape).toEqual([1, 1, 2, 2, 2]);
    expectTensorValues(channelLast.toTypedArray(), [1, 101, 2, 102, 3, 103, 4, 104]);
    expectTensorValues(roundTrip.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("applies left-only temporal padding for causal Conv3d", () => {
    using conv = new QwenImageCausalConv3d(1, 1, [3, 1, 1], 1, [1, 0, 0], 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([10, 20, 30], [1, 3, 1, 1, 1]);
    initialWeight.free();
    using input = MxArray.fromData([5], [1, 1, 1, 1, 1]);
    using output = conv.forward(input);

    mxEval(output);
    expect(output.shape).toEqual([1, 1, 1, 1, 1]);
    expectTensorValues(output.toTypedArray(), [150]);
  });

  test("exposes causal Conv3d configuration for checkpoint-shape validation", () => {
    using conv = new QwenImageCausalConv3d(2, 3, [1, 2, 3], [2, 1, 1], [0, 1, 2], [1, 2, 1]);

    expect(conv.inputChannels).toBe(2);
    expect(conv.outputChannels).toBe(3);
    expect(conv.kernelSize).toEqual([1, 2, 3]);
    expect(conv.stride).toEqual([2, 1, 1]);
    expect(conv.padding).toEqual([0, 1, 2]);
    expect(conv.dilation).toEqual([1, 2, 1]);
    expect(conv.bias).not.toBeNull();
  });

  test("reduces causal padding by provided prefix frames", () => {
    using conv = new QwenImageCausalConv3d(1, 1, [3, 1, 1], 1, [1, 0, 0], 1, false);
    const initialWeight = conv.weight;
    conv.weight = MxArray.fromData([10, 20, 30], [1, 3, 1, 1, 1]);
    initialWeight.free();
    using input = MxArray.fromData([5], [1, 1, 1, 1, 1]);
    using prefix = MxArray.fromData([7], [1, 1, 1, 1, 1]);
    using output = conv.forward(input, prefix);

    mxEval(output);
    expect(output.shape).toEqual([1, 1, 1, 1, 1]);
    expectTensorValues(output.toTypedArray(), [290]);
  });

  test("resamples spatial dimensions while keeping the frame axis explicit", () => {
    using upsample = new QwenImageResample(4, "upsample2d");
    using downsample = new QwenImageResample(4, "downsample2d");
    using upsampleInput = ones([1, 1, 2, 2, 4]);
    using downsampleInput = ones([1, 1, 4, 4, 4]);
    using upsampled = upsample.forward(upsampleInput);
    using downsampled = downsample.forward(downsampleInput);

    mxEval(upsampled, downsampled);
    expect(upsampled.shape).toEqual([1, 1, 4, 4, 2]);
    expect(downsampled.shape).toEqual([1, 1, 2, 2, 4]);
  });

  test("constructs temporal resamplers with explicit time convolutions", () => {
    using upsample = new QwenImageResample(4, "upsample3d");
    using downsample = new QwenImageResample(4, "downsample3d");
    using identity = new QwenImageResample(4, "none");
    using input = ones([1, 1, 2, 2, 4]);
    using retained = identity.forward(input);

    expect(upsample.mode).toBe("upsample3d");
    expect(upsample.timeConv).not.toBeNull();
    expect(downsample.mode).toBe("downsample3d");
    expect(downsample.timeConv).not.toBeNull();
    expect(identity.mode).toBe("none");
    expect(identity.timeConv).toBeNull();
    expect(retained.shape).toEqual(input.shape);
  });

  test("rejects malformed causal prefix shapes", () => {
    using conv = new QwenImageCausalConv3d(1, 1, [3, 1, 1], 1, [1, 0, 0]);
    using input = ones([1, 1, 1, 1, 1]);
    using badPrefix = ones([1, 1, 2, 1, 1]);

    expect(() => conv.forward(input, badPrefix)).toThrow("prefix shape");
  });
});
