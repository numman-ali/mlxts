import { describe, expect, test } from "bun:test";
import { add, MxArray, mxEval, zeros } from "@mlxts/core";

import {
  decodeLtxVideoLatents,
  denormalizeLtxVideoLatents,
  type LtxVideoLatentDecoder,
} from "./decoding";
import { packLtxVideoLatents } from "./latents";

class RecordingLtxVideoDecoder implements LtxVideoLatentDecoder {
  readonly latentChannels = 2;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly scalingFactor: number;
  readonly inputs: number[][] = [];
  readonly mode: "identity" | "zeros";

  constructor(
    options: {
      latentsMean?: readonly number[];
      latentsStd?: readonly number[];
      scalingFactor?: number;
      mode?: "identity" | "zeros";
    } = {},
  ) {
    this.latentsMean = options.latentsMean ?? [0, 0];
    this.latentsStd = options.latentsStd ?? [1, 1];
    this.scalingFactor = options.scalingFactor ?? 1;
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

describe("LTX video latent decoding", () => {
  test("decodes packed latents into BFHWC video output", () => {
    const decoder = new RecordingLtxVideoDecoder();
    using latents = MxArray.fromData([-1, -0.5, 0, 0.5, 0.25, -0.25, 0.75, -0.75], [1, 2, 1, 2, 2]);
    using packed = packLtxVideoLatents(latents);
    using video = decodeLtxVideoLatents(decoder, packed, 1, 2, 2);

    mxEval(video);
    expect(video.shape).toEqual([1, 1, 2, 2, 2]);
    expectCloseList(video.toTypedArray(), [0, 0.625, 0.25, 0.375, 0.5, 0.875, 0.75, 0.125]);
  });

  test("applies per-channel LTX mean, std, and scaling before VAE decode", () => {
    const decoder = new RecordingLtxVideoDecoder({
      latentsMean: [1, 10],
      latentsStd: [2, 3],
      scalingFactor: 2,
      mode: "zeros",
    });
    using latents = MxArray.fromData([0.5, 1, 1.5, 2, 3, 4, 5, 6], [1, 2, 1, 2, 2]);
    using packed = packLtxVideoLatents(latents);
    using video = decodeLtxVideoLatents(decoder, packed, 1, 2, 2);

    mxEval(video);
    expect(video.shape).toEqual([1, 1, 2, 2, 2]);
    expectCloseList(decoder.inputs[0] ?? [], [1.5, 2, 2.5, 3, 14.5, 16, 17.5, 19]);
    expectCloseList(video.toTypedArray(), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  test("rejects latent statistics that do not match the decoder channel count", () => {
    const decoder = new RecordingLtxVideoDecoder({ latentsMean: [0], latentsStd: [1] });
    using latents = MxArray.fromData([0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 2]);
    using packed = packLtxVideoLatents(latents);

    expect(() => decodeLtxVideoLatents(decoder, packed, 1, 2, 2)).toThrow("mean/std");
  });

  test("rejects non-finite latent stats and invalid scaling", () => {
    using latents = zeros([1, 2, 1, 1, 1]);

    expect(() =>
      denormalizeLtxVideoLatents(
        new RecordingLtxVideoDecoder({ latentsMean: [Number.NaN, 0] }),
        latents,
      ),
    ).toThrow("mean values");
    expect(() =>
      denormalizeLtxVideoLatents(
        new RecordingLtxVideoDecoder({ latentsStd: [1, Number.POSITIVE_INFINITY] }),
        latents,
      ),
    ).toThrow("std values");
    expect(() =>
      denormalizeLtxVideoLatents(new RecordingLtxVideoDecoder({ scalingFactor: 0 }), latents),
    ).toThrow("scalingFactor");
  });

  test("rejects decoder outputs outside the BCFHW sample contract", () => {
    const decoder: LtxVideoLatentDecoder = {
      latentChannels: 2,
      latentsMean: [0, 0],
      latentsStd: [1, 1],
      scalingFactor: 1,
      decodeRaw: () => zeros([1, 2, 3, 4]),
    };
    using latents = zeros([1, 2, 1, 1, 1]);
    using packed = packLtxVideoLatents(latents);

    expect(() => decodeLtxVideoLatents(decoder, packed, 1, 1, 1)).toThrow("decoded BCFHW");
  });
});
