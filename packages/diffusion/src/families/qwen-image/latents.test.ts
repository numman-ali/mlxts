import { describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createQwenImageInitialLatents,
  packQwenImageLatents,
  qwenImageLatentShape,
  qwenImagePackedLatentShape,
  qwenImageRopeImageShape,
  qwenImageRopeImageShapeFromLatents,
  unpackQwenImageLatents,
} from "./latents";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Qwen-Image latent helpers", () => {
  test("computes Qwen-Image latent and packed sequence shapes", () => {
    expect(
      qwenImageLatentShape({
        batchSize: 2,
        height: 256,
        width: 384,
        latentChannels: 16,
        vaeScaleFactor: 8,
        patchSize: 2,
      }),
    ).toEqual([2, 16, 1, 32, 48]);
    expect(qwenImagePackedLatentShape(2, 32, 48, 16)).toEqual([2, 384, 64]);
    expect(() =>
      qwenImageLatentShape({
        batchSize: 1,
        height: 250,
        width: 384,
        latentChannels: 16,
        vaeScaleFactor: 8,
        patchSize: 2,
      }),
    ).toThrow("height");
  });

  test("packs NCFHW latent images into Diffusers-compatible 2x2 patch sequences", () => {
    using latents = MxArray.fromData(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [1, 1, 1, 4, 4],
    );
    using packed = packQwenImageLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 4, 4]);
    expectTensorValues(
      packed.toTypedArray(),
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
    );
  });

  test("keeps Diffusers patch channel order for multi-channel latents", () => {
    using latents = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 1, 8]);
    expectTensorValues(packed.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("accepts already-squeezed NCHW latents for pachifier parity", () => {
    using latents = MxArray.fromData(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [1, 1, 4, 4],
    );
    using packed = packQwenImageLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 4, 4]);
    expectTensorValues(
      packed.toTypedArray(),
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
    );
  });

  test("unpacks Qwen-Image patch sequences back to NCFHW latent images", () => {
    using packed = MxArray.fromData(
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
      [1, 4, 4],
    );
    using unpacked = unpackQwenImageLatents(packed, 4, 4);

    unpacked.eval();
    expect(unpacked.shape).toEqual([1, 1, 1, 4, 4]);
    expectTensorValues(
      unpacked.toTypedArray(),
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
  });

  test("creates packed scheduler initial latents", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using packed = createQwenImageInitialLatents({
      scheduler,
      batchSize: 2,
      height: 32,
      width: 32,
      latentChannels: 4,
      vaeScaleFactor: 8,
      dtype: "float16",
    });

    expect(packed.shape).toEqual([2, 4, 16]);
    expect(packed.dtype).toBe("float16");
  });

  test("derives Qwen-Image RoPE image shape from the latent patch grid", () => {
    expect(
      qwenImageRopeImageShape({
        batchSize: 1,
        height: 1024,
        width: 768,
        latentChannels: 16,
        vaeScaleFactor: 8,
        patchSize: 2,
      }),
    ).toEqual([1, 64, 48]);

    using latents = MxArray.fromData([0, 0, 0, 0], [1, 1, 1, 2, 2]);
    expect(qwenImageRopeImageShapeFromLatents(latents)).toEqual([1, 1, 1]);
  });

  test("rejects malformed latent shapes", () => {
    using oddHeight = MxArray.fromData([1, 2], [1, 1, 1, 1, 2]);
    using multiFrame = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2]);
    using rankThree = MxArray.fromData([1, 2], [1, 2, 1]);
    using rankTwoPacked = MxArray.fromData([1, 2, 3, 4], [1, 4]);
    using badPackedChannels = MxArray.fromData([1, 2], [1, 1, 2]);
    using packed = MxArray.fromData([1, 2, 3, 4], [1, 1, 4]);

    expect(() => qwenImagePackedLatentShape(0, 2, 2, 1)).toThrow("batchSize");
    expect(() => qwenImagePackedLatentShape(1, 3, 2, 1)).toThrow("latentHeight");
    expect(() => packQwenImageLatents(oddHeight)).toThrow("height");
    expect(() => packQwenImageLatents(multiFrame)).toThrow("single frame");
    expect(() => packQwenImageLatents(rankThree)).toThrow("NCHW");
    expect(() => unpackQwenImageLatents(rankTwoPacked, 2, 2)).toThrow("rank 3");
    expect(() => unpackQwenImageLatents(badPackedChannels, 2, 2)).toThrow("packedChannels");
    expect(() => unpackQwenImageLatents(packed, 4, 4)).toThrow("patches");
  });
});
