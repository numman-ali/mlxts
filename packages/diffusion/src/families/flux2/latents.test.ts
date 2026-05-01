import { describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createFlux2InitialLatents,
  createFlux2LatentIds,
  createFlux2TextIds,
  flux2LatentMapShape,
  flux2PackedLatentShape,
  packFlux2Latents,
  patchifyFlux2VaeLatents,
  unpackFlux2Latents,
  unpatchifyFlux2VaeLatents,
} from "./latents";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("FLUX.2 Klein latent helpers", () => {
  test("computes Diffusers-compatible packed latent map and sequence shapes", () => {
    expect(flux2LatentMapShape(1, 16, 32, 2)).toEqual([1, 8, 1, 2]);
    expect(flux2PackedLatentShape(1, 16, 32, 2)).toEqual([1, 2, 8]);
    expect(() => flux2LatentMapShape(1, 15, 32, 2)).toThrow("height");
  });

  test("patchifies NCHW VAE latents into FLUX.2 2x2 packed maps", () => {
    using latents = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 2, 2]);
    using packedMap = patchifyFlux2VaeLatents(latents);

    packedMap.eval();
    expect(packedMap.shape).toEqual([1, 8, 1, 1]);
    expectTensorValues(packedMap.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("unpatchifies FLUX.2 packed maps back into NCHW VAE latents", () => {
    using packedMap = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 8, 1, 1]);
    using latents = unpatchifyFlux2VaeLatents(packedMap);

    latents.eval();
    expect(latents.shape).toEqual([1, 2, 2, 2]);
    expectTensorValues(latents.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("packs and unpacks FLUX.2 latent maps as row-major token sequences", () => {
    using latentMap = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 2, 2]);
    using packed = packFlux2Latents(latentMap);
    using unpacked = unpackFlux2Latents(packed, 2, 2);

    packed.eval();
    unpacked.eval();
    expect(packed.shape).toEqual([1, 4, 2]);
    expectTensorValues(packed.toTypedArray(), [1, 101, 2, 102, 3, 103, 4, 104]);
    expectTensorValues(unpacked.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("creates 4-axis text and latent position ids", () => {
    using textIds = createFlux2TextIds(3);
    using latentIds = createFlux2LatentIds(2, 3);

    textIds.eval();
    latentIds.eval();
    expect(textIds.shape).toEqual([3, 4]);
    expect(latentIds.shape).toEqual([6, 4]);
    expectTensorValues(textIds.toTypedArray(), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2]);
    expectTensorValues(
      latentIds.toTypedArray(),
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0],
    );
  });

  test("samples packed initial latents through the scheduler", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using latents = createFlux2InitialLatents({
      scheduler,
      batchSize: 1,
      height: 16,
      width: 16,
      latentChannels: 2,
      dtype: "float32",
    });

    expect(latents.shape).toEqual([1, 1, 8]);
    expect(latents.dtype).toBe("float32");
  });

  test("rejects malformed FLUX.2 latent helper inputs", () => {
    using rankThree = MxArray.fromData([1, 2], [1, 2, 1]);
    using oddHeight = MxArray.fromData([1, 2], [1, 1, 1, 2]);
    using badPacked = MxArray.fromData([1, 2], [1, 1, 2]);

    expect(() => packFlux2Latents(rankThree)).toThrow("NCHW");
    expect(() => patchifyFlux2VaeLatents(oddHeight)).toThrow("height");
    expect(() => unpackFlux2Latents(badPacked, 2, 2)).toThrow("tokens");
    expect(() => createFlux2TextIds(0)).toThrow("sequenceLength");
  });
});
