import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";

import { StableDiffusion3PatchEmbed, stableDiffusion3TimestepEmbedding } from "./embeddings";

describe("stableDiffusion3TimestepEmbedding", () => {
  test("creates cosine-sine FlowMatch timestep embeddings", () => {
    using timesteps = array([0, 500], "float32");
    using embedding = stableDiffusion3TimestepEmbedding(timesteps, 8, "float32");

    expect(embedding.shape).toEqual([2, 8]);
    expect(Array.from(embedding.toTypedArray()).every(Number.isFinite)).toBe(true);
  });
});

describe("StableDiffusion3PatchEmbed", () => {
  test("projects NHWC latents into patch sequences with center-cropped sincos positions", () => {
    using embedder = new StableDiffusion3PatchEmbed({
      sampleSize: 4,
      patchSize: 2,
      inChannels: 4,
      hiddenSize: 8,
      posEmbedMaxSize: 4,
    });
    using latents = zeros([1, 4, 4, 4]);
    using sequence = embedder.forward(latents);

    expect(sequence.shape).toEqual([1, 4, 8]);
    expect(Array.from(sequence.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("rejects latent grids beyond the configured positional crop", () => {
    using embedder = new StableDiffusion3PatchEmbed({
      sampleSize: 4,
      patchSize: 2,
      inChannels: 4,
      hiddenSize: 8,
      posEmbedMaxSize: 2,
    });
    using latents = zeros([1, 8, 4, 4]);

    expect(() => embedder.forward(latents)).toThrow("posEmbedMaxSize");
  });
});
