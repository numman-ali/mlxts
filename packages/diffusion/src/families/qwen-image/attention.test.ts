import { describe, expect, test } from "bun:test";
import { array, mxEval, reshape } from "@mlxts/core";

import { applyQwenImageRotary, QwenImageJointAttention } from "./attention";
import { QwenImageRopeEmbedder } from "./embeddings";

describe("Qwen-Image joint attention", () => {
  test("applies RoPE matrices to attention tensors", () => {
    using queries = array(
      [
        [
          [
            [1, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 2, 2],
          ],
        ],
      ],
      "float32",
    );
    using identityRope = array(
      [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
      "float32",
    );
    using matrices = reshape(identityRope, [1, 1, 2, 3, 2, 2]);
    using rotated = applyQwenImageRotary(queries, matrices);

    mxEval(rotated);
    expect(rotated.shape).toEqual([1, 1, 2, 6]);
  });

  test("runs joint non-causal attention with a non-contiguous text mask", () => {
    using attention = new QwenImageJointAttention(12, 2, 6);
    using image = array(
      [
        [
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
          [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
      ],
      "float32",
    );
    using text = array(
      [
        [
          [0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
          [1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.2],
        ],
      ],
      "float32",
    );
    using mask = array([[1, 0, 1]], "bool");
    using rope = new QwenImageRopeEmbedder(6, 10000, [2, 2, 2]);
    using matrices = rope.embed([1, 1, 2], 3, "float32");
    const output = attention.run(image, text, matrices, mask);
    try {
      mxEval(output.image, output.text);
      expect(output.image.shape).toEqual([1, 2, 12]);
      expect(output.text.shape).toEqual([1, 3, 12]);
    } finally {
      output.image.free();
      output.text.free();
    }
  });
});
