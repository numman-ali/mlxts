import { describe, expect, test } from "bun:test";

import { array, MxArray } from "@mlxts/core";

import {
  createCausalMask,
  createFastAttentionMask,
  createLeftPaddedAttentionMask,
  createStepAttentionMask,
  createStepCausalMask,
} from "./masks";

describe("causal masks", () => {
  test("creates masks in the requested dtype for fused attention compatibility", () => {
    using mask = createCausalMask(2, 4, 2, "bfloat16");
    expect(mask).not.toBeNull();
    expect(mask?.dtype).toBe("bfloat16");
    expect(mask?.shape).toEqual([1, 1, 2, 4]);
  });

  test("returns null for the single-token incremental decode fast path", () => {
    expect(createCausalMask(1, 5, 4, "bfloat16")).toBeNull();
  });

  test("returns null for trimmed single-token sliding decode when the cache already bounds visibility", () => {
    expect(createStepAttentionMask(1, 2048, 512, true)).toBeNull();
  });

  test("sizes trimmed sliding prefill masks from the retained cache span", () => {
    const mask = createStepAttentionMask(3, 4, 4, true);
    expect(mask).toBeInstanceOf(MxArray);
    if (!(mask instanceof MxArray)) {
      throw new Error("expected a tensor mask");
    }
    using ownedMask = mask;
    expect(ownedMask.shape).toEqual([1, 1, 3, 6]);
  });

  test("supports sliding-window causal masks for local attention layers", () => {
    using mask = createCausalMask(3, 3, 0, "float32", 2);
    expect(mask?.toList()).toEqual([
      [
        [
          [0, -1e9, -1e9],
          [0, 0, -1e9],
          [-1e9, 0, 0],
        ],
      ],
    ]);
  });

  test("creates a trimmed additive sliding mask for multi-token prefill", () => {
    using mask = createStepCausalMask(2, 4, "float32", 3, true);
    expect(mask).toBeInstanceOf(MxArray);
    if (!(mask instanceof MxArray)) {
      throw new Error("expected a tensor mask");
    }
    expect(mask.shape).toEqual([1, 1, 2, 4]);
    expect(mask.toList()).toEqual([
      [
        [
          [0, 0, 0, -1e9],
          [-1e9, 0, 0, 0],
        ],
      ],
    ]);
  });

  test("createFastAttentionMask returns the fused fast-path markers when available", () => {
    expect(createFastAttentionMask(2, 2, 0)).toBe("causal");
    expect(createFastAttentionMask(2, 4, 2)).toBe("causal");
  });

  test("createFastAttentionMask keeps explicit masks for cached sliding-window prefill", () => {
    const mask = createFastAttentionMask(2, 4, 2, 3);
    expect(mask).toBeInstanceOf(MxArray);
    if (!(mask instanceof MxArray)) {
      throw new Error("expected a tensor mask");
    }
    using ownedMask = mask;
    expect(ownedMask.dtype).toBe("bool");
    expect(ownedMask.shape).toEqual([1, 1, 2, 4]);
    expect(ownedMask.toList()).toEqual([
      [
        [
          [1, 1, 1, 0],
          [0, 1, 1, 1],
        ],
      ],
    ]);
  });

  test("creates batched causal masks for left-padded prompts", () => {
    using leftPadding = array([1, 3, 0], "int32");
    using mask = createLeftPaddedAttentionMask(4, 4, 0, leftPadding);

    expect(mask.dtype).toBe("bool");
    expect(mask.shape).toEqual([3, 1, 4, 4]);
    expect(mask.toList()).toEqual([
      [
        [
          [0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 1, 1, 0],
          [0, 1, 1, 1],
        ],
      ],
      [
        [
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1],
        ],
      ],
      [
        [
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [1, 1, 1, 1],
        ],
      ],
    ]);
  });

  test("creates batched left-padded decode masks against retained prompt state", () => {
    using leftPadding = array([1, 3], "int32");
    using mask = createLeftPaddedAttentionMask(1, 5, 4, leftPadding);

    expect(mask.shape).toEqual([2, 1, 1, 5]);
    expect(mask.toList()).toEqual([[[[0, 1, 1, 1, 1]]], [[[0, 0, 0, 1, 1]]]]);
  });

  test("creates left-padded sliding masks after stale positions are trimmed", () => {
    using leftPadding = array([1, 0], "int32");
    using mask = createLeftPaddedAttentionMask(2, 4, 2, leftPadding, 3);

    expect(mask.shape).toEqual([2, 1, 2, 4]);
    expect(mask.toList()).toEqual([
      [
        [
          [0, 1, 1, 0],
          [0, 1, 1, 1],
        ],
      ],
      [
        [
          [1, 1, 1, 0],
          [0, 1, 1, 1],
        ],
      ],
    ]);
  });

  test("rejects non-positive query and key lengths", () => {
    expect(() => createCausalMask(0, 1, 0)).toThrow("must be positive");
    expect(() => createCausalMask(1, 0, 0)).toThrow("must be positive");
    expect(() => createFastAttentionMask(0, 1, 0)).toThrow("must be positive");
    expect(() => createFastAttentionMask(1, 0, 0)).toThrow("must be positive");
    using leftPadding = array([0], "int32");
    expect(() => createLeftPaddedAttentionMask(0, 1, 0, leftPadding)).toThrow("must be positive");
    expect(() => createLeftPaddedAttentionMask(1, 0, 0, leftPadding)).toThrow("must be positive");
  });

  test("rejects non-positive sliding-window sizes", () => {
    expect(() => createCausalMask(1, 1, 0, "float32", 0)).toThrow("windowSize");
    expect(() => createFastAttentionMask(1, 1, 0, 0)).toThrow("windowSize");
    using leftPadding = array([0], "int32");
    expect(() => createLeftPaddedAttentionMask(1, 1, 0, leftPadding, 0)).toThrow("windowSize");
  });
});
