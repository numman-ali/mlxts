import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";

import {
  applyVisionRotaryPosEmb,
  bilinearInterpolationTables,
  flattenPairEmbeddings,
  gridGeometry,
  gridThwList,
  reorderedPatchIndices,
  repeatFrames,
  rotaryPairIndices,
  takeAxis1Slice,
  takeSequenceSlice,
} from "./vision-support";

describe("Qwen 3.5 vision support helpers", () => {
  test("parse grids and derive merged patch ordering helpers", () => {
    using gridThw = array([[1, 2, 2]], "int32");
    expect(gridThwList(gridThw, "vision test")).toEqual([[1, 2, 2]]);

    const geometry = gridGeometry([1, 2, 2], 1, "vision test");
    expect(geometry).toEqual({
      frames: 1,
      height: 2,
      width: 2,
      mergedHeight: 2,
      mergedWidth: 2,
      frameSize: 4,
    });
    expect(rotaryPairIndices(geometry, 1)).toEqual([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    expect(reorderedPatchIndices([1, 2, 2], 1, "vision test")).toEqual([0, 1, 2, 3]);
    expect(bilinearInterpolationTables(2, 2, 2)).toEqual({
      indices: [
        [0, 1, 2, 3],
        [1, 1, 3, 3],
        [2, 3, 2, 3],
        [3, 3, 3, 3],
      ],
      weights: [
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
    });

    expect(() => gridGeometry([1, 3, 2], 2, "vision test")).toThrow(
      "must be divisible by spatial merge size 2",
    );
  });

  test("slice helpers and embedding reshaping validate tensor layouts", () => {
    using sequence = array(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      "float32",
    );
    using sequenceSlice = takeSequenceSlice(sequence, 1, 1, "vision test");
    expect(sequenceSlice.toList()).toEqual([
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    expect(() => takeSequenceSlice(sequence, 2, 1, "vision test")).toThrow(
      "cannot slice range [2, 3)",
    );

    using qkv = array(
      [
        [[[1, 2]], [[3, 4]], [[5, 6]]],
        [[[7, 8]], [[9, 10]], [[11, 12]]],
      ],
      "float32",
    );
    using axisSlice = takeAxis1Slice(qkv, 1, "vision test");
    expect(axisSlice.toList()).toEqual([[[3, 4]], [[9, 10]]]);
    expect(() => takeAxis1Slice(qkv, 3, "vision test")).toThrow("expected [seq, axis, heads, dim]");

    using pairEmbeddings = array(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      "float32",
    );
    using flattened = flattenPairEmbeddings(pairEmbeddings, 2, "vision test");
    expect(flattened.toList()).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    using repeated = repeatFrames(flattened, 2);
    expect(repeated.toList()).toEqual([
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [5, 6, 7, 8],
    ]);
  });

  test("rotary application preserves queries and keys when sin is zero", () => {
    using queries = array([[[1, 2, 3, 4]]], "float32");
    using keys = array([[[4, 3, 2, 1]]], "float32");
    using cosEmbeddings = array([[1, 1, 1, 1]], "float32");
    using sinEmbeddings = zeros([1, 4], "float32");
    const rotated = applyVisionRotaryPosEmb(queries, keys, cosEmbeddings, sinEmbeddings);

    try {
      expect(rotated.queries.toList()).toEqual([[[1, 2, 3, 4]]]);
      expect(rotated.keys.toList()).toEqual([[[4, 3, 2, 1]]]);
    } finally {
      rotated.queries.free();
      rotated.keys.free();
    }
  });
});
