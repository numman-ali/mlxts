import { describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createLtx2AudioInitialLatents,
  createLtxVideoInitialLatents,
  ltx2AudioLatentLength,
  ltx2AudioLatentShape,
  ltx2AudioPackedLatentShape,
  ltxVideoLatentShape,
  ltxVideoPackedLatentShape,
  packLtx2AudioLatents,
  packLtxVideoLatents,
  unpackLtx2AudioLatents,
  unpackLtxVideoLatents,
} from "./latents";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("LTX latent helpers", () => {
  test("computes LTX video latent and packed sequence shapes", () => {
    expect(
      ltxVideoLatentShape({
        batchSize: 1,
        height: 512,
        width: 704,
        numFrames: 161,
        latentChannels: 128,
      }),
    ).toEqual([1, 128, 21, 16, 22]);
    expect(
      ltxVideoLatentShape({
        batchSize: 1,
        height: 512,
        width: 704,
        numFrames: 160,
        latentChannels: 128,
      }),
    ).toEqual([1, 128, 20, 16, 22]);
    expect(
      ltxVideoLatentShape({
        batchSize: 1,
        height: 512,
        width: 704,
        numFrames: 121,
        latentChannels: 128,
      }),
    ).toEqual([1, 128, 16, 16, 22]);
    expect(ltxVideoPackedLatentShape(2, 4, 8, 10, 16, 2, 2)).toEqual([2, 40, 128]);
    expect(() =>
      ltxVideoLatentShape({
        batchSize: 1,
        height: 513,
        width: 704,
        numFrames: 161,
        latentChannels: 128,
      }),
    ).toThrow("height");
  });

  test("packs BCFHW LTX video latents into Diffusers token order", () => {
    using latents = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 1, 2, 2]);
    using packed = packLtxVideoLatents(latents);

    packed.eval();
    expect(packed.shape).toEqual([1, 4, 2]);
    expectTensorValues(packed.toTypedArray(), [1, 101, 2, 102, 3, 103, 4, 104]);
  });

  test("packs and unpacks patched LTX video latents", () => {
    using latents = MxArray.fromData(
      [1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 103, 104, 105, 106, 107, 108],
      [1, 2, 2, 2, 2],
    );
    using packed = packLtxVideoLatents(latents, 2, 1);
    using unpacked = unpackLtxVideoLatents(packed, 2, 2, 2, 2, 1);

    packed.eval();
    unpacked.eval();
    expect(packed.shape).toEqual([1, 2, 8]);
    expectTensorValues(
      packed.toTypedArray(),
      [1, 2, 3, 4, 101, 102, 103, 104, 5, 6, 7, 8, 105, 106, 107, 108],
    );
    expectTensorValues(
      unpacked.toTypedArray(),
      [1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 103, 104, 105, 106, 107, 108],
    );
  });

  test("creates packed scheduler video latents", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using latents = createLtxVideoInitialLatents({
      scheduler,
      batchSize: 1,
      height: 64,
      width: 64,
      numFrames: 9,
      latentChannels: 4,
      dtype: "float16",
    });

    expect(latents.shape).toEqual([1, 8, 4]);
    expect(latents.dtype).toBe("float16");
  });

  test("computes LTX-2 audio latent and packed sequence shapes", () => {
    expect(ltx2AudioLatentLength(161, 24)).toBe(168);
    expect(ltx2AudioLatentLength(12, 24)).toBe(12);
    expect(
      ltx2AudioLatentShape({
        batchSize: 1,
        numFrames: 161,
        frameRate: 24,
        latentChannels: 8,
        melBins: 64,
      }),
    ).toEqual([1, 8, 168, 16]);
    expect(ltx2AudioPackedLatentShape(2, 168, 16, 8)).toEqual([2, 168, 128]);
    expect(ltx2AudioPackedLatentShape(2, 168, 16, 8, 2, 4)).toEqual([2, 336, 64]);
    expect(() => ltx2AudioPackedLatentShape(1, 168, 16, 8, 2)).toThrow("both");
  });

  test("packs BCLM LTX-2 audio latents into Diffusers token order", () => {
    using latents = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 2, 2]);
    using packed = packLtx2AudioLatents(latents);
    using unpacked = unpackLtx2AudioLatents(packed, 2, 2);

    packed.eval();
    unpacked.eval();
    expect(packed.shape).toEqual([1, 2, 4]);
    expectTensorValues(packed.toTypedArray(), [1, 2, 101, 102, 3, 4, 103, 104]);
    expectTensorValues(unpacked.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("packs and unpacks patched LTX-2 audio latents", () => {
    using latents = MxArray.fromData([1, 2, 3, 4, 101, 102, 103, 104], [1, 2, 2, 2]);
    using packed = packLtx2AudioLatents(latents, 2, 1);
    using unpacked = unpackLtx2AudioLatents(packed, 2, 2, 2, 1);

    packed.eval();
    unpacked.eval();
    expect(packed.shape).toEqual([1, 2, 4]);
    expectTensorValues(packed.toTypedArray(), [1, 2, 101, 102, 3, 4, 103, 104]);
    expectTensorValues(unpacked.toTypedArray(), [1, 2, 3, 4, 101, 102, 103, 104]);
  });

  test("creates packed scheduler audio latents", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using latents = createLtx2AudioInitialLatents({
      scheduler,
      batchSize: 1,
      numFrames: 24,
      frameRate: 24,
      latentChannels: 8,
      melBins: 64,
      dtype: "float32",
    });

    expect(latents.shape).toEqual([1, 25, 128]);
    expect(latents.dtype).toBe("float32");

    using patched = createLtx2AudioInitialLatents({
      scheduler,
      batchSize: 1,
      numFrames: 24,
      frameRate: 24,
      latentChannels: 8,
      melBins: 64,
      patchSize: 2,
      patchSizeT: 1,
      dtype: "float16",
    });

    expect(patched.shape).toEqual([1, 200, 16]);
    expect(patched.dtype).toBe("float16");
  });

  test("rejects malformed LTX latent helper inputs", () => {
    using badVideo = MxArray.fromData([1, 2], [1, 1, 1, 1, 2]);
    using rankThreeVideo = MxArray.fromData([1, 2], [1, 2, 1]);
    using badVideoPacked = MxArray.fromData([1, 2, 3, 4], [1, 2, 2]);
    using badAudio = MxArray.fromData([1, 2, 3], [1, 1, 1, 3]);
    using badAudioPacked = MxArray.fromData([1, 2, 3, 4, 5, 6], [1, 2, 3]);

    expect(() => packLtxVideoLatents(badVideo, 2)).toThrow("height");
    expect(() => packLtxVideoLatents(rankThreeVideo)).toThrow("BCFHW");
    expect(() => unpackLtxVideoLatents(badVideoPacked, 2, 2, 2, 2, 1)).toThrow("packedChannels");
    expect(() => packLtx2AudioLatents(badAudio, 2, 1)).toThrow("melBins");
    expect(() => unpackLtx2AudioLatents(badAudioPacked, 2, 2)).toThrow("packedChannels");
  });
});
