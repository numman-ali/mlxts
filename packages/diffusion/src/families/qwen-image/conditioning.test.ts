import { describe, expect, test } from "bun:test";
import { add, MxArray, mxEval } from "@mlxts/core";

import {
  encodeQwenImageLatents,
  prepareQwenImageReferenceLatents,
  type QwenImageLatentEncoder,
} from "./conditioning";

class QueuedQwenImageEncoder implements QwenImageLatentEncoder {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly sampleShapes: number[][] = [];
  #latents: MxArray[];

  constructor(
    latents: readonly MxArray[],
    options: { latentsMean?: readonly number[]; latentsStd?: readonly number[] } = {},
  ) {
    this.#latents = [...latents];
    this.latentChannels = this.#latents[0]?.shape[1] ?? 0;
    this.latentsMean = options.latentsMean ?? Array.from({ length: this.latentChannels }, () => 0);
    this.latentsStd = options.latentsStd ?? Array.from({ length: this.latentChannels }, () => 1);
  }

  encodeRaw(sample: MxArray): MxArray {
    this.sampleShapes.push([...sample.shape]);
    const latents = this.#latents.shift();
    if (latents === undefined) {
      throw new Error("QueuedQwenImageEncoder: missing queued latents.");
    }
    return add(latents, 0);
  }
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("Qwen-Image image conditioning latents", () => {
  test("encodes raw VAE latents into normalized packed reference latents", () => {
    using rawLatents = MxArray.fromData([2, 4, 6, 8, 13, 16, 19, 22], [1, 2, 1, 2, 2]);
    using sample = MxArray.fromData([0, 0, 0, 0], [1, 1, 1, 2, 2]);
    const encoder = new QueuedQwenImageEncoder([rawLatents], {
      latentsMean: [1, 10],
      latentsStd: [2, 3],
    });

    using packed = encodeQwenImageLatents(encoder, sample);

    mxEval(packed);
    expect(packed.shape).toEqual([1, 1, 8]);
    expectCloseList(packed.toTypedArray(), [0.5, 1.5, 2.5, 3.5, 1, 2, 3, 4]);
    expect(encoder.sampleShapes).toEqual([[1, 1, 1, 2, 2]]);
  });

  test("concatenates multiple reference latent segments and records RoPE shapes", () => {
    using firstLatents = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 2, 2]);
    using secondLatents = MxArray.fromData([5, 6, 7, 8], [1, 1, 1, 2, 2]);
    using firstSample = MxArray.fromData([0], [1, 1, 1, 1, 1]);
    using secondSample = MxArray.fromData([1], [1, 1, 1, 1, 1]);
    const encoder = new QueuedQwenImageEncoder([firstLatents, secondLatents]);

    const prepared = prepareQwenImageReferenceLatents(encoder, [firstSample, secondSample]);
    using packed = prepared.packedLatents;

    mxEval(packed);
    expect(packed.shape).toEqual([1, 2, 4]);
    expect(prepared.imageShapes).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
    expectCloseList(packed.toTypedArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  test("rejects malformed reference preparation inputs", () => {
    using firstLatents = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 2, 2]);
    using secondLatents = MxArray.fromData([5, 6, 7, 8, 9, 10, 11, 12], [2, 1, 1, 2, 2]);
    using sample = MxArray.fromData([0], [1, 1, 1, 1, 1]);

    expect(() => prepareQwenImageReferenceLatents(new QueuedQwenImageEncoder([]), [])).toThrow(
      "at least one",
    );
    expect(() =>
      encodeQwenImageLatents(
        new QueuedQwenImageEncoder([firstLatents], { latentsMean: [], latentsStd: [] }),
        sample,
      ),
    ).toThrow("mean/std");
    expect(() =>
      prepareQwenImageReferenceLatents(new QueuedQwenImageEncoder([firstLatents, secondLatents]), [
        sample,
        sample,
      ]),
    ).toThrow("batches");
  });
});
