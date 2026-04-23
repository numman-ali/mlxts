import { describe, expect, test } from "bun:test";
import {
  array,
  clearMemoryCache,
  getActiveMemoryBytes,
  mxEval,
  random,
  synchronize,
} from "@mlxts/core";

import { GPT_TINY, resolveConfig } from "../config";
import { CausalSelfAttention } from "./causal-self-attention";

const TEST_CONFIG = resolveConfig(
  {
    ...GPT_TINY,
    nLayer: 2,
    nHead: 2,
    nEmbd: 32,
    blockSize: 16,
    dropout: 0.2,
  },
  26,
);

function maxRange(values: readonly number[]): number {
  return Math.max(...values) - Math.min(...values);
}

describe("CausalSelfAttention", () => {
  test("repeated training-mode forward passes stay within a bounded active-memory range", () => {
    random.seed(42);
    const attention = new CausalSelfAttention(TEST_CONFIG);
    attention.train();

    using input = array(
      Array.from({ length: 1 }, () =>
        Array.from({ length: 8 }, () => Array.from({ length: TEST_CONFIG.nEmbd }, () => 1)),
      ),
      "float32",
    );

    clearMemoryCache();
    synchronize();
    const activeSamples: number[] = [];

    try {
      for (let iteration = 0; iteration < 12; iteration++) {
        using output = attention.forward(input);
        mxEval(output);
        synchronize();

        if (iteration >= 3) {
          activeSamples.push(getActiveMemoryBytes());
        }
      }

      expect(activeSamples.length).toBeGreaterThan(0);
      expect(maxRange(activeSamples)).toBeLessThan(8 * 1024 * 1024);
    } finally {
      attention[Symbol.dispose]();
    }
  });
});
