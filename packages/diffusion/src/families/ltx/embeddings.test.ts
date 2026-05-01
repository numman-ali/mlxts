import { describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";

import {
  createLtx2AudioCoords,
  createLtx2RotaryEmbeddings,
  createLtx2VideoCoords,
  createLtxVideoRopeCoords,
  createLtxVideoRotaryEmbeddings,
  type LtxRotaryEmbeddings,
} from "./embeddings";

function expectClose(actual: number | undefined, expected: number, digits = 6): void {
  expect(actual).toBeDefined();
  expect(actual ?? Number.NaN).toBeCloseTo(expected, digits);
}

function freeRotaryEmbeddings(embeddings: LtxRotaryEmbeddings): void {
  embeddings.cos.free();
  embeddings.sin.free();
}

describe("LTX rotary and coordinate helpers", () => {
  test("creates classic LTX video coordinates in Diffusers token order", () => {
    using coords = createLtxVideoRopeCoords({
      batchSize: 2,
      latentFrames: 2,
      latentHeight: 2,
      latentWidth: 2,
      ropeInterpolationScale: [8 / 24, 32, 32],
    });

    coords.eval();
    expect(coords.shape).toEqual([2, 8, 3]);
    const values = coords.toTypedArray();
    expectClose(Number(values[0]), 0);
    expectClose(Number(values[1]), 0);
    expectClose(Number(values[2]), 0);
    expectClose(Number(values[5]), 32 / 2048);
    expectClose(Number(values[12]), 1 / 60);
    expectClose(Number(values[24]), 0);
    expectClose(Number(values[26]), 0);
  });

  test("creates classic LTX interleaved RoPE frequencies", () => {
    const embeddings = createLtxVideoRotaryEmbeddings({
      batchSize: 1,
      latentFrames: 1,
      latentHeight: 1,
      latentWidth: 1,
      dim: 8,
    });
    try {
      embeddings.cos.eval();
      embeddings.sin.eval();
      expect(embeddings.cos.shape).toEqual([1, 1, 8]);
      expect(embeddings.sin.shape).toEqual([1, 1, 8]);
      const cosValues = embeddings.cos.toTypedArray();
      const sinValues = embeddings.sin.toTypedArray();
      expectClose(Number(cosValues[0]), 1);
      expectClose(Number(cosValues[1]), 1);
      for (let index = 2; index < 8; index += 1) {
        expectClose(Number(cosValues[index]), 0);
        expectClose(Number(sinValues[index]), -1);
      }
    } finally {
      freeRotaryEmbeddings(embeddings);
    }
  });

  test("creates LTX-2 video patch-boundary coordinates", () => {
    using coords = createLtx2VideoCoords({
      batchSize: 1,
      latentFrames: 2,
      latentHeight: 2,
      latentWidth: 2,
      frameRate: 24,
    });

    coords.eval();
    expect(coords.shape).toEqual([1, 3, 8, 2]);
    const values = coords.toTypedArray();
    expectClose(Number(values[0]), 0);
    expectClose(Number(values[1]), 1 / 24);
    expectClose(Number(values[8]), 1 / 24);
    expectClose(Number(values[9]), 9 / 24);
    expectClose(Number(values[16]), 0);
    expectClose(Number(values[17]), 32);
    expectClose(Number(values[32]), 0);
    expectClose(Number(values[33]), 32);
  });

  test("creates LTX-2 audio patch-boundary coordinates", () => {
    using coords = createLtx2AudioCoords({
      batchSize: 2,
      audioLatentFrames: 3,
      patchSizeT: 2,
      shift: 1,
    });

    coords.eval();
    expect(coords.shape).toEqual([2, 1, 2, 2]);
    const values = coords.toTypedArray();
    expectClose(Number(values[0]), 0.01);
    expectClose(Number(values[1]), 0.09);
    expectClose(Number(values[2]), 0.09);
    expectClose(Number(values[3]), 0.17);
    expectClose(Number(values[4]), 0.01);
    expectClose(Number(values[5]), 0.09);
  });

  test("creates LTX-2 interleaved RoPE from patch-boundary coordinates", () => {
    using coords = MxArray.fromData([0, 0], [1, 1, 1, 2], "float32");
    const embeddings = createLtx2RotaryEmbeddings({
      coords,
      dim: 4,
      modality: "audio",
      ropeType: "interleaved",
      baseNumFrames: 20,
      theta: 4,
    });
    try {
      embeddings.cos.eval();
      embeddings.sin.eval();
      expect(embeddings.cos.shape).toEqual([1, 1, 4]);
      expect(embeddings.sin.shape).toEqual([1, 1, 4]);
      const cosValues = embeddings.cos.toTypedArray();
      const sinValues = embeddings.sin.toTypedArray();
      expectClose(Number(cosValues[0]), 0);
      expectClose(Number(cosValues[1]), 0);
      expectClose(Number(cosValues[2]), 1);
      expectClose(Number(cosValues[3]), 1);
      expectClose(Number(sinValues[0]), -1);
      expectClose(Number(sinValues[1]), -1);
      expectClose(Number(sinValues[2]), 0);
      expectClose(Number(sinValues[3]), 0);
    } finally {
      freeRotaryEmbeddings(embeddings);
    }
  });

  test("creates LTX-2 split RoPE in attention-head layout", () => {
    using coords = MxArray.fromData([0, 0, 10, 10, 20, 20], [1, 3, 1, 2], "float32");
    const embeddings = createLtx2RotaryEmbeddings({
      coords,
      dim: 16,
      modality: "video",
      ropeType: "split",
      baseNumFrames: 20,
      baseHeight: 20,
      baseWidth: 20,
      numAttentionHeads: 2,
      theta: 1,
    });
    try {
      embeddings.cos.eval();
      embeddings.sin.eval();
      expect(embeddings.cos.shape).toEqual([1, 2, 1, 4]);
      expect(embeddings.sin.shape).toEqual([1, 2, 1, 4]);
      const cosValues = embeddings.cos.toTypedArray();
      const sinValues = embeddings.sin.toTypedArray();
      const expectedCos = [1, 1, 0, 1, 0, 0, 1, 0];
      const expectedSin = [0, 0, -1, 0, 1, -1, 0, 1];
      for (let index = 0; index < expectedCos.length; index += 1) {
        expectClose(Number(cosValues[index]), expectedCos[index] ?? Number.NaN);
        expectClose(Number(sinValues[index]), expectedSin[index] ?? Number.NaN);
      }
    } finally {
      freeRotaryEmbeddings(embeddings);
    }
  });

  test("rejects malformed LTX coordinate and RoPE inputs", () => {
    using badCoords = MxArray.fromData([0, 1, 2, 3], [1, 2, 1, 2], "float32");
    expect(() =>
      createLtx2VideoCoords({
        batchSize: 1,
        latentFrames: 1,
        latentHeight: 3,
        latentWidth: 2,
        patchSize: 2,
      }),
    ).toThrow("latentHeight");
    expect(() =>
      createLtx2RotaryEmbeddings({
        coords: badCoords,
        dim: 4,
        modality: "audio",
      }),
    ).toThrow("audio");
    const badOptions: Parameters<typeof createLtx2RotaryEmbeddings>[0] = {
      coords: MxArray.fromData([0, 0, 0, 0, 0, 0], [1, 3, 1, 2], "float32"),
      dim: 4,
      modality: "video",
    };
    try {
      Object.assign(badOptions, { ropeType: "spiral" });
      expect(() => createLtx2RotaryEmbeddings(badOptions)).toThrow("ropeType");
    } finally {
      badOptions.coords.free();
    }
  });
});
