import { describe, expect, test } from "bun:test";

import { array } from "@mlxts/core";
import { disposePreparedPrompt, slicePreparedPrompt } from "./prepared-prompt";

describe("prepared prompt helpers", () => {
  test("slices prompt tokens, embeddings, and rank-2 position ids together", () => {
    using inputEmbeddings = array([[[0], [1], [2], [3]]], "float32");
    using positionIds = array([[10, 11, 12, 13]], "int32");
    const sliced = slicePreparedPrompt(
      { tokenIds: [0, 1, 2, 3], inputEmbeddings, positionIds },
      2,
      2,
    );

    try {
      expect(sliced.tokenIds).toEqual([2, 3]);
      expect(sliced.inputEmbeddings?.toList()).toEqual([[[2], [3]]]);
      expect(sliced.positionIds?.toList()).toEqual([[12, 13]]);
    } finally {
      disposePreparedPrompt(sliced);
    }
  });

  test("slices rank-3 position ids used by Qwen image prompts", () => {
    using positionIds = array([[[0, 1, 2, 3]], [[10, 11, 12, 13]], [[20, 21, 22, 23]]], "int32");
    const sliced = slicePreparedPrompt({ tokenIds: [0, 1, 2, 3], positionIds }, 1, 2);

    try {
      expect(sliced.tokenIds).toEqual([1, 2]);
      expect(sliced.positionIds?.toList()).toEqual([[[1, 2]], [[11, 12]], [[21, 22]]]);
    } finally {
      disposePreparedPrompt(sliced);
    }
  });

  test("rejects invalid prepared prompt slices", () => {
    expect(() => slicePreparedPrompt({ tokenIds: [0, 1] }, -1, 1)).toThrow(
      "start must be a non-negative integer",
    );
    expect(() => slicePreparedPrompt({ tokenIds: [0, 1] }, 1, 0)).toThrow(
      "length must be a positive integer",
    );
    expect(() => slicePreparedPrompt({ tokenIds: [0, 1] }, 1, 2)).toThrow("cannot slice [1, 3)");
  });
});
