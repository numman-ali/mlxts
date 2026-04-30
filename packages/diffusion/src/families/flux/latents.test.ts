import { describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";

import {
  createFluxLatentImageIds,
  fluxPackedLatentShape,
  packFluxLatents,
  unpackFluxLatents,
} from "./latents";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Flux latent helpers", () => {
  test("computes packed latent shapes", () => {
    expect(fluxPackedLatentShape(2, 64, 32, 16)).toEqual([2, 512, 64]);
    expect(() => fluxPackedLatentShape(1, 63, 32, 16)).toThrow("latentHeight");
  });

  test("packs NHWC latent images into 2x2 patch sequences", () => {
    using latents = MxArray.fromData(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [1, 4, 4, 1],
    );
    using packed = packFluxLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 4, 4]);
    expectTensorValues(
      packed.toTypedArray(),
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
    );
  });

  test("keeps Diffusers patch channel order for multi-channel latents", () => {
    using latents = MxArray.fromData([1, 101, 2, 102, 3, 103, 4, 104], [1, 2, 2, 2]);
    using packed = packFluxLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 1, 8]);
    expectTensorValues(packed.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("unpacks Flux patch sequences back to NHWC latent images", () => {
    using packed = MxArray.fromData(
      [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
      [1, 4, 4],
    );
    using unpacked = unpackFluxLatents(packed, 4, 4);

    unpacked.eval();
    expect(unpacked.shape).toEqual([1, 4, 4, 1]);
    expectTensorValues(
      unpacked.toTypedArray(),
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
  });

  test("creates Flux image position ids", () => {
    using ids = createFluxLatentImageIds(2, 3);

    ids.eval();
    expect(ids.shape).toEqual([6, 3]);
    expect(ids.dtype).toBe("int32");
    expectTensorValues(ids.toTypedArray(), [0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 2]);
  });

  test("rejects malformed latent shapes", () => {
    using oddHeight = MxArray.fromData([1, 2], [1, 1, 2, 1]);
    using rankThree = MxArray.fromData([1, 2], [1, 2, 1]);
    using packed = MxArray.fromData([1, 2, 3, 4], [1, 1, 4]);

    expect(() => packFluxLatents(oddHeight)).toThrow("height");
    expect(() => packFluxLatents(rankThree)).toThrow("NHWC");
    expect(() => unpackFluxLatents(packed, 4, 4)).toThrow("patches");
  });
});
