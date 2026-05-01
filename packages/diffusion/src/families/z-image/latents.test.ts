import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createZImageCoordinateIds,
  createZImageInitialLatents,
  padZImageFeature,
  patchifyZImageLatent,
  sliceZImageLatentBatchItem,
  stackZImageLatentBatchItems,
  unpatchifyZImageLatent,
  zImageLatentShape,
} from "./latents";

describe("Z-Image latent helpers", () => {
  test("computes Diffusers-compatible latent shape", () => {
    expect(
      zImageLatentShape({
        batchSize: 1,
        height: 256,
        width: 384,
        latentChannels: 16,
        vaeScaleFactor: 8,
      }),
    ).toEqual([1, 16, 32, 48]);
    expect(() =>
      zImageLatentShape({
        batchSize: 1,
        height: 250,
        width: 384,
        latentChannels: 16,
        vaeScaleFactor: 8,
      }),
    ).toThrow("height");
  });

  test("creates initial NCHW noise latents", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using latents = createZImageInitialLatents({
      scheduler,
      batchSize: 1,
      height: 32,
      width: 32,
      latentChannels: 4,
      vaeScaleFactor: 8,
      dtype: "float32",
    });

    expect(latents.shape).toEqual([1, 4, 4, 4]);
  });

  test("builds flattened coordinate ids", () => {
    using ids = createZImageCoordinateIds([1, 2, 2], [5, 7, 11]);

    expect(ids.shape).toEqual([4, 3]);
    expect(ids.toList()).toEqual([
      [5, 7, 11],
      [5, 7, 12],
      [5, 8, 11],
      [5, 8, 12],
    ]);
  });

  test("pads feature rows to the requested sequence multiple", () => {
    using feature = MxArray.fromData([1, 2, 3, 4, 5, 6], [3, 2]);
    const padded = padZImageFeature(feature, 4, [4, 1, 1], [1, 0, 0]);
    try {
      mxEval(padded.features, padded.positionIds, padded.padMask);
      expect(padded.totalLength).toBe(4);
      expect(padded.originalLength).toBe(3);
      expect(padded.features.toList()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6],
        [5, 6],
      ]);
      expect(JSON.stringify(padded.padMask.toList())).toBe("[0,0,0,1]");
    } finally {
      padded.features.free();
      padded.positionIds.free();
      padded.padMask.free();
    }
  });

  test("retains feature rows when no sequence padding is needed", () => {
    using feature = MxArray.fromData([1, 2, 3, 4], [2, 2]);
    const padded = padZImageFeature(feature, 2, [1, 1, 2], [0, 0, 0]);
    try {
      mxEval(padded.features, padded.positionIds, padded.padMask);
      expect(padded.totalLength).toBe(2);
      expect(padded.originalLength).toBe(2);
      expect(padded.features.toList()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(JSON.stringify(padded.padMask.toList())).toBe("[0,0]");
    } finally {
      padded.features.free();
      padded.positionIds.free();
      padded.padMask.free();
    }
  });

  test("round-trips patchified Z-Image latents", () => {
    using latent = MxArray.fromData([1, 2, 3, 4], [1, 1, 2, 2]);
    const geometry = { patchSize: 2, framePatchSize: 1, packedLatentChannels: 4 };
    using patches = patchifyZImageLatent(latent, geometry).patches;
    using roundTripped = unpatchifyZImageLatent(
      patches,
      { frames: 1, height: 2, width: 2 },
      geometry,
      1,
    );

    mxEval(roundTripped);
    expect(roundTripped.shape).toEqual([1, 1, 2, 2]);
    expect(roundTripped.toList()).toEqual([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
  });

  test("rejects patch geometry that does not tile the latent shape", () => {
    using latent = zeros([1, 1, 2, 2]);
    using channelMismatch = zeros([2, 1, 2, 2]);
    using badPatchShape = zeros([2, 3]);
    using shortPatches = zeros([1, 4]);

    expect(() =>
      patchifyZImageLatent(channelMismatch, {
        patchSize: 2,
        framePatchSize: 1,
        packedLatentChannels: 4,
      }),
    ).toThrow("latent channels");
    expect(() =>
      patchifyZImageLatent(latent, { patchSize: 3, framePatchSize: 1, packedLatentChannels: 9 }),
    ).toThrow("height");
    expect(() =>
      unpatchifyZImageLatent(
        badPatchShape,
        { frames: 1, height: 2, width: 2 },
        { patchSize: 2, framePatchSize: 1, packedLatentChannels: 4 },
        1,
      ),
    ).toThrow("expected patch features");
    expect(() =>
      unpatchifyZImageLatent(
        shortPatches,
        { frames: 1, height: 2, width: 4 },
        { patchSize: 2, framePatchSize: 1, packedLatentChannels: 4 },
        1,
      ),
    ).toThrow("expected at least");
  });

  test("slices and stacks latent batch items", () => {
    using latents = MxArray.fromData(
      Array.from({ length: 8 }, (_, index) => index + 1),
      [2, 1, 2, 2],
    );
    using first = sliceZImageLatentBatchItem(latents, 0);
    using second = sliceZImageLatentBatchItem(latents, 1);
    using stacked = stackZImageLatentBatchItems([first, second]);

    mxEval(first, second, stacked);
    expect(first.shape).toEqual([1, 1, 2, 2]);
    expect(second.shape).toEqual([1, 1, 2, 2]);
    expect(stacked.shape).toEqual([2, 1, 2, 2]);
    expect(stacked.toList()).toEqual(latents.toList());
  });

  test("rejects invalid latent batch slicing and stacking", () => {
    using latents = zeros([1, 1, 2, 2]);
    using rank3 = zeros([1, 2, 2]);
    using multiFrame = zeros([1, 2, 2, 2]);

    expect(() => sliceZImageLatentBatchItem(rank3, 0)).toThrow("NCHW");
    expect(() => sliceZImageLatentBatchItem(latents, 2)).toThrow("out of range");
    expect(() => stackZImageLatentBatchItems([])).toThrow("at least one sample");
    expect(() => stackZImageLatentBatchItems([multiFrame])).toThrow("frame dimension");
  });
});
