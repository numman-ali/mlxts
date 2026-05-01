import { describe, expect, test } from "bun:test";
import { mxEval, zeros } from "@mlxts/core";

import { StableDiffusion3JointAttention, StableDiffusion3SelfAttention } from "./attention";

describe("Stable Diffusion 3 attention", () => {
  test("runs context-pre-only joint attention without a context output projection", () => {
    using attention = new StableDiffusion3JointAttention({
      hiddenSize: 8,
      numHeads: 2,
      headDim: 4,
      qkNorm: "rms_norm",
      contextPreOnly: true,
    });
    using hiddenStates = zeros([1, 2, 8]);
    using encoderHiddenStates = zeros([1, 3, 8]);

    const output = attention.run(hiddenStates, encoderHiddenStates);
    try {
      mxEval(output.hidden);
      expect(output.hidden.shape).toEqual([1, 2, 8]);
      expect(output.context).toBeNull();
    } finally {
      output.hidden.free();
      output.context?.free();
    }
  });

  test("guards unsupported forward and malformed attention shapes", () => {
    using attention = new StableDiffusion3JointAttention({
      hiddenSize: 8,
      numHeads: 2,
      headDim: 4,
      qkNorm: null,
      contextPreOnly: false,
    });
    using selfAttention = new StableDiffusion3SelfAttention(8, 2, 4, null);
    using hiddenStates = zeros([1, 2, 8]);
    using wrongContext = zeros([2, 3, 8]);
    using wrongRank = zeros([1, 2]);

    expect(() => attention.forward(hiddenStates)).toThrow("use run");
    expect(() => attention.run(hiddenStates, wrongContext)).toThrow("context shape");
    expect(() => selfAttention.forward(wrongRank)).toThrow("rank-3");
    expect(
      () =>
        new StableDiffusion3JointAttention({
          hiddenSize: 7,
          numHeads: 2,
          headDim: 4,
          qkNorm: null,
          contextPreOnly: false,
        }),
    ).toThrow("hiddenSize");
    expect(() => new StableDiffusion3SelfAttention(7, 2, 4, null)).toThrow("hiddenSize");
  });
});
