import { describe, expect, test } from "bun:test";

import { MxArray } from "@mlxts/core";

import { splitPackedQueryGateHeads } from "./attention";

describe("Qwen3_5TextAttention helpers", () => {
  test("splits packed q/gate projections per head instead of by one global midpoint", () => {
    using packed = MxArray.fromData([1, 2, 101, 102, 3, 4, 103, 104], [1, 1, 8]);
    const split = splitPackedQueryGateHeads(packed, 2, 2);
    try {
      expect(split.queries.shape).toEqual([1, 1, 4]);
      expect(split.gates.shape).toEqual([1, 1, 4]);
      expect(split.queries.toList()).toEqual([[[1, 2, 3, 4]]]);
      expect(split.gates.toList()).toEqual([[[101, 102, 103, 104]]]);
    } finally {
      split.queries.free();
      split.gates.free();
    }
  });

  test("rejects packed projections whose last dimension does not match the configured heads", () => {
    using packed = MxArray.fromData([1, 2, 3, 4], [1, 1, 4]);
    expect(() => splitPackedQueryGateHeads(packed, 2, 2)).toThrow("expected last dimension 8");
  });
});
