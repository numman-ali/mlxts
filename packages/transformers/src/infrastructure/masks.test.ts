import { describe, expect, test } from "bun:test";

import { MxArray } from "@mlxts/core";

import {
  createCausalMask,
  createFastAttentionMask,
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

    const mask = createFastAttentionMask(2, 4, 2);
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
          [1, 1, 1, 1],
        ],
      ],
    ]);
  });

  test("rejects non-positive query and key lengths", () => {
    expect(() => createCausalMask(0, 1, 0)).toThrow("must be positive");
    expect(() => createCausalMask(1, 0, 0)).toThrow("must be positive");
    expect(() => createFastAttentionMask(0, 1, 0)).toThrow("must be positive");
    expect(() => createFastAttentionMask(1, 0, 0)).toThrow("must be positive");
  });

  test("rejects non-positive sliding-window sizes", () => {
    expect(() => createCausalMask(1, 1, 0, "float32", 0)).toThrow("windowSize");
    expect(() => createFastAttentionMask(1, 1, 0, 0)).toThrow("windowSize");
  });
});
