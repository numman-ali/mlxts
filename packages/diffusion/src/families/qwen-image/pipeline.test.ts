import { describe, expect, test } from "bun:test";
import { add, MxArray, mxEval, zeros } from "@mlxts/core";

import { packQwenImageLatents } from "./latents";
import { decodeQwenImageLatents, type QwenImageLatentDecoder } from "./pipeline";

class RecordingQwenImageDecoder implements QwenImageLatentDecoder {
  readonly latentChannels = 2;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly inputs: number[][] = [];
  readonly mode: "identity" | "zeros";

  constructor(
    options: {
      latentsMean?: readonly number[];
      latentsStd?: readonly number[];
      mode?: "identity" | "zeros";
    } = {},
  ) {
    this.latentsMean = options.latentsMean ?? [0, 0];
    this.latentsStd = options.latentsStd ?? [1, 1];
    this.mode = options.mode ?? "identity";
  }

  decodeRaw(latents: MxArray): MxArray {
    mxEval(latents);
    this.inputs.push(Array.from(latents.toTypedArray()));
    if (this.mode === "identity") {
      return add(latents, 0);
    }
    return zeros([...latents.shape], latents.dtype);
  }
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("Qwen-Image latent decoding", () => {
  test("decodes packed latents through single-frame VAE output and NHWC postprocess", () => {
    const decoder = new RecordingQwenImageDecoder();
    using latents = MxArray.fromData([-1, -0.5, 0, 0.5, 0.25, -0.25, 0.75, -0.75], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);
    using decoded = decodeQwenImageLatents(decoder, packed, 2, 2);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(decoded.toTypedArray(), [0, 0.625, 0.25, 0.375, 0.5, 0.875, 0.75, 0.125]);
  });

  test("applies Qwen-Image per-channel mean and std before raw VAE decode", () => {
    const decoder = new RecordingQwenImageDecoder({
      latentsMean: [1, 10],
      latentsStd: [2, 3],
      mode: "zeros",
    });
    using latents = MxArray.fromData([0.5, 1, 1.5, 2, 3, 4, 5, 6], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);
    using decoded = decodeQwenImageLatents(decoder, packed, 2, 2);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(decoder.inputs[0] ?? [], [2, 3, 4, 5, 19, 22, 25, 28]);
    expectCloseList(decoded.toTypedArray(), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  test("rejects VAE latent statistics that do not match latent channels", () => {
    const decoder = new RecordingQwenImageDecoder({ latentsMean: [0], latentsStd: [1] });
    using latents = MxArray.fromData([0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);

    expect(() => decodeQwenImageLatents(decoder, packed, 2, 2)).toThrow("mean/std");
  });
});
