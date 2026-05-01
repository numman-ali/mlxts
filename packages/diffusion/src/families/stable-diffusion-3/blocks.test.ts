import { describe, expect, test } from "bun:test";
import { zeros } from "@mlxts/core";

import { StableDiffusion3JointTransformerBlock } from "./blocks";

describe("Stable Diffusion 3 transformer blocks", () => {
  test("keeps block execution on the explicit image/context run surface", () => {
    using block = new StableDiffusion3JointTransformerBlock({
      hiddenSize: 8,
      numHeads: 2,
      headDim: 4,
      qkNorm: null,
      contextPreOnly: false,
      useDualAttention: false,
    });
    using hiddenStates = zeros([1, 2, 8]);

    expect(() => block.forward(hiddenStates)).toThrow("use run");
  });
});
