import { describe, expect, test } from "bun:test";
import { clearMemoryCache, getActiveMemoryBytes, mxEval, ones, zeros } from "@mlxts/core";

import { gatedDeltaSequence } from "./gated-delta";

describe("Qwen3_5 gated delta attention", () => {
  test("releases the local recurrent state handle after returning the retained final state", () => {
    clearMemoryCache();
    const beforeBytes = getActiveMemoryBytes();

    using q = ones([1, 1, 64, 64], "float32");
    using k = ones([1, 1, 64, 64], "float32");
    using v = ones([1, 1, 64, 64], "float32");
    using g = ones([1, 1, 64], "float32");
    using beta = ones([1, 1, 64], "float32");
    using initialState = zeros([1, 64, 64, 64], "float32");

    for (let index = 0; index < 4; index += 1) {
      const result = gatedDeltaSequence(q, k, v, g, beta, initialState);
      try {
        mxEval(result.output, result.state);
      } finally {
        result.output.free();
        result.state.free();
      }
    }

    clearMemoryCache();
    const afterBytes = getActiveMemoryBytes();
    expect(afterBytes - beforeBytes).toBeLessThan(8 * 1024 * 1024);
  });
});
