import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import { QwenImageRopeEmbedder, qwenImageTimestepEmbedding } from "./embeddings";

describe("Qwen-Image transformer embeddings", () => {
  test("creates Diffusers-style timestep embeddings", () => {
    using timesteps = array([0.25, 0.5], "float32");
    using embedded = qwenImageTimestepEmbedding(timesteps, 8);

    embedded.eval();
    expect(embedded.shape).toEqual([2, 8]);
    expect(Array.from(embedded.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("builds combined text/image RoPE matrices with scaled image positions", () => {
    using rope = new QwenImageRopeEmbedder(6, 10000, [2, 2, 2]);
    using embedded = rope.embed([1, 1, 2], 3, "float32");

    mxEval(embedded);
    expect(embedded.shape).toEqual([1, 1, 5, 3, 2, 2]);
    expect(Array.from(embedded.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("rejects invalid RoPE geometry", () => {
    expect(() => new QwenImageRopeEmbedder(8, 10000, [2, 2, 2])).toThrow("headDim");
    using rope = new QwenImageRopeEmbedder(6, 10000, [2, 2, 2]);
    expect(() => rope.embed([1, 0, 2], 3, "float32")).toThrow("image shape");
    expect(() => rope.embed([1, 1, 2], 0, "float32")).toThrow("textLength");
  });
});
