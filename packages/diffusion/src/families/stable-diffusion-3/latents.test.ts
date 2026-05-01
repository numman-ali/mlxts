import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createStableDiffusion3InitialLatents,
  stableDiffusion3LatentShape,
  unpatchifyStableDiffusion3Latents,
} from "./latents";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("stableDiffusion3LatentShape", () => {
  test("returns NHWC latent dimensions from image dimensions", () => {
    expect(
      stableDiffusion3LatentShape({
        batchSize: 2,
        height: 1024,
        width: 512,
        latentChannels: 16,
        vaeScaleFactor: 8,
      }),
    ).toEqual([2, 128, 64, 16]);
  });

  test("rejects invalid latent geometry", () => {
    expect(() =>
      stableDiffusion3LatentShape({
        batchSize: 0,
        height: 1024,
        width: 512,
        latentChannels: 16,
      }),
    ).toThrow("batchSize");
    expect(() =>
      stableDiffusion3LatentShape({
        batchSize: 1,
        height: 1025,
        width: 512,
        latentChannels: 16,
      }),
    ).toThrow("height");
  });

  test("creates scheduler-scaled initial NHWC latents", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });

    using latents = createStableDiffusion3InitialLatents({
      scheduler,
      batchSize: 1,
      height: 8,
      width: 8,
      latentChannels: 4,
    });

    mxEval(latents);
    expect(latents.shape).toEqual([1, 1, 1, 4]);
    expect(Array.from(latents.toTypedArray()).every(Number.isFinite)).toBe(true);
  });
});

describe("unpatchifyStableDiffusion3Latents", () => {
  test("restores row-major patch tokens into NHWC latents", () => {
    using patches = array(
      [
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16],
        ],
      ],
      "float32",
    );
    using latents = unpatchifyStableDiffusion3Latents(patches, 4, 4, 2, 1);

    expect(latents.shape).toEqual([1, 4, 4, 1]);
    expectTensorValues(
      latents.toTypedArray(),
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
    );
  });

  test("rejects malformed patch geometry", () => {
    using patches = array([[[1, 2, 3, 4]]], "float32");

    expect(() => unpatchifyStableDiffusion3Latents(patches, 3, 4, 2, 1)).toThrow("latentHeight");
    expect(() => unpatchifyStableDiffusion3Latents(patches, 4, 4, 2, 1)).toThrow(
      "expected patches",
    );
  });
});
