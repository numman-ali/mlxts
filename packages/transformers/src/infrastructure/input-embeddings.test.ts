import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";

import {
  retainInputEmbeddings,
  retainInputPositionIds,
  retainPromptInputEmbeddings,
  retainPromptPositionIds,
  slicePromptInputEmbeddings,
  slicePromptPositionIds,
  validatePromptInputEmbeddings,
  validatePromptPositionIds,
} from "./input-embeddings";

describe("input-embedding helpers", () => {
  test("retainInputEmbeddings validates rank, sequence alignment, and hidden size", () => {
    using inputIds = array([[1, 2]], "int32");
    using inputEmbeddings = array(
      [
        [
          [10, 11],
          [20, 21],
        ],
      ],
      "float32",
    );
    const retained = retainInputEmbeddings(inputIds, inputEmbeddings, 2, "test");
    if (retained === null) {
      throw new Error("expected retained embeddings");
    }
    using ownedRetained = retained;
    expect(ownedRetained.toList()).toEqual([
      [
        [10, 11],
        [20, 21],
      ],
    ]);

    using wrongHidden = array([[[10], [20]]], "float32");
    expect(() => retainInputEmbeddings(inputIds, wrongHidden, 2, "test")).toThrow(
      "hidden size 1 must match model hidden size 2",
    );
    expect(retainInputEmbeddings(inputIds, undefined, 2, "test")).toBeNull();

    using badSequence = array([[[10, 11]]], "float32");
    expect(() => retainInputEmbeddings(inputIds, badSequence, 2, "test")).toThrow(
      "must match token ids",
    );
  });

  test("retainPromptInputEmbeddings validates and retains prompt-aligned embeddings", () => {
    using promptEmbeddings = array(
      [
        [
          [10, 11],
          [20, 21],
        ],
      ],
      "float32",
    );
    const retained = retainPromptInputEmbeddings([1, 2], promptEmbeddings, "prompt test");
    if (retained === null) {
      throw new Error("expected retained prompt embeddings");
    }
    using ownedRetained = retained;
    expect(ownedRetained.toList()).toEqual([
      [
        [10, 11],
        [20, 21],
      ],
    ]);

    using badBatch = array(
      [
        [
          [10, 11],
          [20, 21],
        ],
        [
          [30, 31],
          [40, 41],
        ],
      ],
      "float32",
    );
    expect(() => retainPromptInputEmbeddings([1, 2], badBatch, "prompt test")).toThrow(
      "currently requires batch size 1",
    );
    expect(retainPromptInputEmbeddings([1, 2], undefined, "prompt test")).toBeNull();

    using badLength = array(
      [
        [
          [10, 11],
          [20, 21],
          [30, 31],
        ],
      ],
      "float32",
    );
    expect(() => retainPromptInputEmbeddings([1, 2], badLength, "prompt test")).toThrow(
      "must match prompt length 2",
    );
  });

  test("slicePromptInputEmbeddings returns the requested prompt window", () => {
    using promptEmbeddings = array(
      [
        [
          [10, 11],
          [20, 21],
          [30, 31],
        ],
      ],
      "float32",
    );
    using slice = slicePromptInputEmbeddings(promptEmbeddings, 1, 2, "slice test");
    expect(slice.toList()).toEqual([
      [
        [20, 21],
        [30, 31],
      ],
    ]);

    using wrongRank = array([[10, 11, 12]], "float32");
    expect(() => slicePromptInputEmbeddings(wrongRank, 0, 1, "slice test")).toThrow(
      "must have shape [batch, seq, hidden]",
    );
  });

  test("retainInputPositionIds accepts token-aligned rank-2 and rank-3 inputs", () => {
    using inputIds = array([[1, 2, 3]], "int32");
    using rank2 = array([[0, 1, 2]], "int32");
    const retainedRank2 = retainInputPositionIds(inputIds, rank2, "position test");
    if (retainedRank2 === null) {
      throw new Error("expected retained rank-2 position ids");
    }
    using ownedRank2 = retainedRank2;
    expect(ownedRank2.toList()).toEqual([[0, 1, 2]]);

    using rank3 = array([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], "int32");
    const retainedRank3 = retainInputPositionIds(inputIds, rank3, "position test");
    if (retainedRank3 === null) {
      throw new Error("expected retained rank-3 position ids");
    }
    using ownedRank3 = retainedRank3;
    expect(ownedRank3.toList()).toEqual([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]]);
    expect(retainInputPositionIds(inputIds, undefined, "position test")).toBeNull();

    using badRank2 = array([[0, 1]], "int32");
    expect(() => retainInputPositionIds(inputIds, badRank2, "position test")).toThrow(
      "must match token ids",
    );

    using badRank4 = array([[[[0, 1, 2]]]], "int32");
    expect(() => retainInputPositionIds(inputIds, badRank4, "position test")).toThrow(
      "must have shape [batch, seq] or [axes, batch, seq]",
    );

    using badInputIds = array([1, 2, 3], "int32");
    expect(() => retainInputPositionIds(badInputIds, rank2, "position test")).toThrow(
      "requires token ids with shape [batch, seq]",
    );
  });

  test("retainPromptPositionIds validates prompt-aligned rank-2 and rank-3 inputs", () => {
    using rank2 = array([[0, 1, 2]], "int32");
    const retainedRank2 = retainPromptPositionIds([1, 2, 3], rank2, "prompt position test");
    if (retainedRank2 === null) {
      throw new Error("expected retained rank-2 prompt position ids");
    }
    using ownedRank2 = retainedRank2;
    expect(ownedRank2.toList()).toEqual([[0, 1, 2]]);

    using rank3 = array([[[0, 1, 2]], [[0, 1, 2]]], "int32");
    const retainedRank3 = retainPromptPositionIds([1, 2, 3], rank3, "prompt position test");
    if (retainedRank3 === null) {
      throw new Error("expected retained rank-3 prompt position ids");
    }
    using ownedRank3 = retainedRank3;
    expect(ownedRank3.toList()).toEqual([[[0, 1, 2]], [[0, 1, 2]]]);
    expect(retainPromptPositionIds([1, 2, 3], undefined, "prompt position test")).toBeNull();

    using badBatch = array(
      [
        [0, 1, 2],
        [0, 1, 2],
      ],
      "int32",
    );
    expect(() => retainPromptPositionIds([1, 2, 3], badBatch, "prompt position test")).toThrow(
      "must be [1, 3]",
    );

    using badRank = array([[[[0, 1, 2]]]], "int32");
    expect(() => retainPromptPositionIds([1, 2, 3], badRank, "prompt position test")).toThrow(
      "must have shape [1, seq] or [axes, 1, seq]",
    );
  });

  test("slicePromptPositionIds slices both rank-2 and rank-3 prompt position ids", () => {
    using rank2 = array([[0, 1, 2, 3]], "int32");
    using rank2Slice = slicePromptPositionIds(rank2, 1, 2, "slice position test");
    expect(rank2Slice.toList()).toEqual([[1, 2]]);

    using rank3 = array([[[0, 1, 2, 3]], [[10, 11, 12, 13]]], "int32");
    using rank3Slice = slicePromptPositionIds(rank3, 1, 2, "slice position test");
    expect(rank3Slice.toList()).toEqual([[[1, 2]], [[11, 12]]]);

    using badBatch = array(
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
      "int32",
    );
    expect(() => slicePromptPositionIds(badBatch, 0, 1, "slice position test")).toThrow(
      "requires batch size 1",
    );

    using badRank = array([0, 1, 2], "int32");
    expect(() => slicePromptPositionIds(badRank, 0, 1, "slice position test")).toThrow(
      "must have shape [1, seq] or [axes, 1, seq]",
    );
  });

  test("prompt validators reject malformed prompt-aligned tensors", () => {
    using wrongRankEmbeddings = array([[10, 11]], "float32");
    expect(() =>
      validatePromptInputEmbeddings([1], wrongRankEmbeddings, "prompt validate"),
    ).toThrow("must have shape [batch, seq, hidden]");

    using wrongLengthEmbeddings = array(
      [
        [
          [10, 11],
          [20, 21],
        ],
      ],
      "float32",
    );
    expect(() =>
      validatePromptInputEmbeddings([1], wrongLengthEmbeddings, "prompt validate"),
    ).toThrow("must match prompt length 1");

    using wrongRankPositionIds = array([0, 1], "int32");
    expect(() =>
      validatePromptPositionIds([1, 2], wrongRankPositionIds, "prompt validate"),
    ).toThrow("must have shape [1, seq] or [axes, 1, seq]");

    using wrongBatchPositionIds = array(
      [
        [
          [0, 1],
          [0, 1],
        ],
      ],
      "int32",
    );
    expect(() =>
      validatePromptPositionIds([1, 2], wrongBatchPositionIds, "prompt validate"),
    ).toThrow("must have batch size 1 and sequence length 2");
  });
});
