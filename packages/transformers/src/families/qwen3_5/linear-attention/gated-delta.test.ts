import { describe, expect, test } from "bun:test";
import {
  array,
  clearMemoryCache,
  concatenate,
  getActiveMemoryBytes,
  mxEval,
  ones,
  repeat,
  slice,
  zeros,
} from "@mlxts/core";

import { gatedDeltaSequence, gatedDeltaSequenceFromKeyHeads } from "./gated-delta";

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error(`flattenNumbers: expected number or array, got ${typeof value}.`);
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

function expectCloseLists(actual: unknown, expected: unknown): void {
  const actualValues = flattenNumbers(actual);
  const expectedValues = flattenNumbers(expected);
  expect(actualValues).toHaveLength(expectedValues.length);
  for (let index = 0; index < actualValues.length; index += 1) {
    expect(actualValues[index]).toBeCloseTo(expectedValues[index] ?? 0, 5);
  }
}

describe("Qwen3_5 gated delta attention", () => {
  test("computes a simple single-step recurrent update", () => {
    using q = ones([1, 1, 1, 2], "float32");
    using k = ones([1, 1, 1, 2], "float32");
    using v = ones([1, 1, 1, 2], "float32");
    using g = ones([1, 1, 1], "float32");
    using beta = ones([1, 1, 1], "float32");
    using initialState = zeros([1, 1, 2, 2], "float32");

    const result = gatedDeltaSequence(q, k, v, g, beta, initialState);
    try {
      expect(result.output.shape).toEqual([1, 1, 1, 2]);
      expect(result.state.shape).toEqual([1, 1, 2, 2]);
      expect(result.output.toList()).toEqual([[[[2, 2]]]]);
      expect(result.state.toList()).toEqual([
        [
          [
            [1, 1],
            [1, 1],
          ],
        ],
      ]);
    } finally {
      result.output.free();
      result.state.free();
    }
  });

  test("native key-head path matches the TypeScript recurrence oracle", () => {
    using q = array(
      [
        [
          [Array.from({ length: 32 }, (_, index) => (index + 1) / 64)],
          [Array.from({ length: 32 }, (_, index) => (index + 2) / 80)],
        ],
      ],
      "float32",
    );
    using k = array(
      [
        [
          [Array.from({ length: 32 }, (_, index) => (index + 3) / 96)],
          [Array.from({ length: 32 }, (_, index) => (index + 4) / 112)],
        ],
      ],
      "float32",
    );
    using v = array(
      [
        [
          [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
          ],
          [
            [0.2, 0.1, 0.4],
            [0.6, 0.3, 0.5],
          ],
        ],
      ],
      "float32",
    );
    using g = array(
      [
        [
          [0.95, 0.9],
          [0.85, 0.8],
        ],
      ],
      "float32",
    );
    using beta = array(
      [
        [
          [0.7, 0.6],
          [0.5, 0.4],
        ],
      ],
      "float32",
    );
    using initialState = zeros([1, 2, 3, 32], "float32");
    using repeatedQ = repeat(q, 2, 2);
    using repeatedK = repeat(k, 2, 2);

    const expected = gatedDeltaSequence(repeatedQ, repeatedK, v, g, beta, initialState);
    const actual = gatedDeltaSequenceFromKeyHeads(q, k, v, g, beta, initialState);
    try {
      mxEval(actual.output, actual.state, expected.output, expected.state);
      expect(actual.output.shape).toEqual([1, 2, 2, 3]);
      expect(actual.state.shape).toEqual([1, 2, 3, 32]);
      expectCloseLists(actual.output.toList(), expected.output.toList());
      expectCloseLists(actual.state.toList(), expected.state.toList());
    } finally {
      actual.output.free();
      actual.state.free();
      expected.output.free();
      expected.state.free();
    }
  });

  test("native key-head path can continue decode state token by token", () => {
    using q = array(
      [
        [
          [Array.from({ length: 32 }, (_, index) => (index + 1) / 70)],
          [Array.from({ length: 32 }, (_, index) => (index + 5) / 90)],
        ],
      ],
      "float32",
    );
    using k = array(
      [
        [
          [Array.from({ length: 32 }, (_, index) => (index + 2) / 75)],
          [Array.from({ length: 32 }, (_, index) => (index + 6) / 95)],
        ],
      ],
      "float32",
    );
    using v = array(
      [
        [
          [
            [0.2, 0.4],
            [0.1, 0.3],
          ],
          [
            [0.5, 0.7],
            [0.6, 0.8],
          ],
        ],
      ],
      "float32",
    );
    using g = array(
      [
        [
          [0.9, 0.85],
          [0.8, 0.75],
        ],
      ],
      "float32",
    );
    using beta = array(
      [
        [
          [0.6, 0.55],
          [0.5, 0.45],
        ],
      ],
      "float32",
    );
    using initialState = zeros([1, 2, 2, 32], "float32");
    using qFirst = slice(q, [0, 0, 0, 0], [1, 1, 1, 32]);
    using kFirst = slice(k, [0, 0, 0, 0], [1, 1, 1, 32]);
    using vFirst = slice(v, [0, 0, 0, 0], [1, 1, 2, 2]);
    using gFirst = slice(g, [0, 0, 0], [1, 1, 2]);
    using betaFirst = slice(beta, [0, 0, 0], [1, 1, 2]);
    using qSecond = slice(q, [0, 1, 0, 0], [1, 2, 1, 32]);
    using kSecond = slice(k, [0, 1, 0, 0], [1, 2, 1, 32]);
    using vSecond = slice(v, [0, 1, 0, 0], [1, 2, 2, 2]);
    using gSecond = slice(g, [0, 1, 0], [1, 2, 2]);
    using betaSecond = slice(beta, [0, 1, 0], [1, 2, 2]);

    const full = gatedDeltaSequenceFromKeyHeads(q, k, v, g, beta, initialState);
    const first = gatedDeltaSequenceFromKeyHeads(
      qFirst,
      kFirst,
      vFirst,
      gFirst,
      betaFirst,
      initialState,
    );
    try {
      const second = gatedDeltaSequenceFromKeyHeads(
        qSecond,
        kSecond,
        vSecond,
        gSecond,
        betaSecond,
        first.state,
      );
      try {
        using stitchedOutput = concatenate([first.output, second.output], 1);
        mxEval(full.output, full.state, stitchedOutput, second.state);
        expectCloseLists(stitchedOutput.toList(), full.output.toList());
        expectCloseLists(second.state.toList(), full.state.toList());
      } finally {
        second.output.free();
        second.state.free();
      }
    } finally {
      full.output.free();
      full.state.free();
      first.output.free();
      first.state.free();
    }
  });

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
