import { describe, expect, test } from "bun:test";
import { mxEval, zeros } from "@mlxts/core";

import {
  disposeStableDiffusion3AdaLayerNormZero,
  disposeStableDiffusion35AdaLayerNormZeroX,
  StableDiffusion3AdaLayerNormContinuous,
  StableDiffusion3AdaLayerNormZero,
  StableDiffusion35AdaLayerNormZeroX,
} from "./normalization";

describe("Stable Diffusion 3 adaptive normalization", () => {
  test("returns forward-compatible hidden states from AdaLN-Zero variants", () => {
    using x = zeros([1, 2, 4]);
    using emb = zeros([1, 4]);
    using norm = new StableDiffusion3AdaLayerNormZero(4);
    using normX = new StableDiffusion35AdaLayerNormZeroX(4);

    using hidden = norm.forward(x, emb);
    using hiddenX = normX.forward(x, emb);

    mxEval(hidden, hiddenX);
    expect(hidden.shape).toEqual([1, 2, 4]);
    expect(hiddenX.shape).toEqual([1, 2, 4]);
  });

  test("exposes all block modulation gates", () => {
    using x = zeros([1, 2, 4]);
    using emb = zeros([1, 4]);
    using norm = new StableDiffusion3AdaLayerNormZero(4);
    using normX = new StableDiffusion35AdaLayerNormZeroX(4);

    const output = norm.modulate(x, emb);
    const outputX = normX.modulate(x, emb);
    try {
      expect(output.hiddenStates.shape).toEqual([1, 2, 4]);
      expect(output.gateMsa.shape).toEqual([1, 1, 4]);
      expect(output.shiftMlp.shape).toEqual([1, 1, 4]);
      expect(outputX.hiddenStates2.shape).toEqual([1, 2, 4]);
      expect(outputX.gateMsa2.shape).toEqual([1, 1, 4]);
    } finally {
      disposeStableDiffusion3AdaLayerNormZero(output);
      disposeStableDiffusion35AdaLayerNormZeroX(outputX);
    }
  });

  test("rejects missing or mismatched modulation inputs", () => {
    using x = zeros([1, 2, 4]);
    using wrongHidden = zeros([1, 2, 5]);
    using emb = zeros([1, 4]);
    using wrongEmb = zeros([1, 5]);
    using wrongBatchEmb = zeros([2, 4]);
    using norm = new StableDiffusion3AdaLayerNormZero(4);
    using normX = new StableDiffusion35AdaLayerNormZeroX(4);
    using continuous = new StableDiffusion3AdaLayerNormContinuous(4, 4);

    expect(() => norm.forward(x)).toThrow("emb is required");
    expect(() => normX.forward(x)).toThrow("emb is required");
    expect(() => norm.modulate(wrongHidden, emb)).toThrow("hidden size");
    expect(() => norm.modulate(x, wrongEmb)).toThrow("emb shape");
    expect(() => continuous.forward(wrongHidden, emb)).toThrow("hidden size");
    expect(() => continuous.forward(x, wrongBatchEmb)).toThrow("conditioning");
  });
});
