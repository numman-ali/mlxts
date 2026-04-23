import { describe, expect, test } from "bun:test";
import type { MxArray } from "./array";
import { array, ones, zeros } from "./array";
import { qwenGatedDeltaUpdate, scaledDotProductAttention } from "./fast";
import { isMetalAvailable } from "./metal";
import { multiply } from "./ops/arithmetic";
import { where } from "./ops/comparison";
import { matmul } from "./ops/linalg";
import { softmax, sum } from "./ops/reduction";
import { transpose, tril } from "./ops/shape";
import { grad, mxEval } from "./transforms";

function manualCausalAttention(
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  scale: number,
): MxArray {
  using maskBase = ones([queries.shape[2] ?? 0, queries.shape[2] ?? 0], "bool");
  using mask = tril(maskBase);
  return manualMaskedAttention(queries, keys, values, scale, mask);
}

function manualMaskedAttention(
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  scale: number,
  mask: MxArray,
): MxArray {
  const sequenceLength = queries.shape[2];
  if (sequenceLength === undefined) {
    throw new Error("manualMaskedAttention: expected rank-4 queries");
  }

  using keyTranspose = transpose(keys, [0, 1, 3, 2]);
  using scores = matmul(queries, keyTranspose);
  using scaledScores = multiply(scores, scale);
  using maskedScores = where(mask, scaledScores, -1e9);
  using weights = softmax(maskedScores, -1);
  return matmul(weights, values);
}

function expectNestedClose(actual: unknown, expected: unknown): void {
  if (typeof actual === "number" && typeof expected === "number") {
    expect(actual).toBeCloseTo(expected, 5);
    return;
  }

  expect(Array.isArray(actual)).toBe(true);
  expect(Array.isArray(expected)).toBe(true);

  const actualItems = actual as unknown[];
  const expectedItems = expected as unknown[];
  expect(actualItems).toHaveLength(expectedItems.length);
  for (let index = 0; index < actualItems.length; index++) {
    expectNestedClose(actualItems[index], expectedItems[index]);
  }
}

describe("fast.scaledDotProductAttention", () => {
  test("causal mode matches the explicit composed implementation", () => {
    const scale = Math.sqrt(1 / 2);
    using queries = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
          ],
        ],
      ],
      "float32",
    );
    using keys = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
          ],
        ],
      ],
      "float32",
    );
    using values = array(
      [
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
        ],
      ],
      "float32",
    );

    using fused = scaledDotProductAttention(queries, keys, values, {
      scale,
      maskMode: "causal",
    });
    using manual = manualCausalAttention(queries, keys, values, scale);

    mxEval(fused, manual);
    expect(fused.shape).toEqual([1, 1, 3, 2]);
    expectNestedClose(fused.toList(), manual.toList());
  });

  test("causal mode matches manual attention for multiple heads and longer sequences", () => {
    const scale = Math.sqrt(1 / 2);
    using queries = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
          ],
          [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 2],
          ],
        ],
      ],
      "float32",
    );
    using keys = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
          ],
          [
            [1, 1],
            [0, 1],
            [1, 0],
            [2, 0],
          ],
        ],
      ],
      "float32",
    );
    using values = array(
      [
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
          ],
          [
            [2, 1],
            [4, 3],
            [6, 5],
            [8, 7],
          ],
        ],
      ],
      "float32",
    );

    using fused = scaledDotProductAttention(queries, keys, values, {
      scale,
      maskMode: "causal",
    });
    using manual = manualCausalAttention(queries, keys, values, scale);

    mxEval(fused, manual);
    expect(fused.shape).toEqual([1, 2, 4, 2]);
    expectNestedClose(fused.toList(), manual.toList());
  });

  test("causal mode aligns shorter query blocks to the end of cached keys", () => {
    const scale = Math.sqrt(1 / 2);
    using queries = array(
      [
        [
          [
            [1, 1],
            [2, 1],
          ],
        ],
      ],
      "float32",
    );
    using keys = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
          ],
        ],
      ],
      "float32",
    );
    using values = array(
      [
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
          ],
        ],
      ],
      "float32",
    );
    using mask = array(
      [
        [
          [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
          ],
        ],
      ],
      "bool",
    );

    using fused = scaledDotProductAttention(queries, keys, values, {
      scale,
      maskMode: "causal",
    });
    using manual = manualMaskedAttention(queries, keys, values, scale, mask);

    mxEval(fused, manual);
    expectNestedClose(fused.toList(), manual.toList());
  });

  test("array mask mode matches the explicit masked implementation", () => {
    const scale = Math.sqrt(1 / 2);
    using queries = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
          ],
        ],
      ],
      "float32",
    );
    using keys = array(
      [
        [
          [
            [1, 0],
            [0, 1],
            [1, 1],
          ],
        ],
      ],
      "float32",
    );
    using values = array(
      [
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
        ],
      ],
      "float32",
    );
    using mask = array(
      [
        [
          [1, 0, 0],
          [1, 1, 0],
          [1, 1, 1],
        ],
      ],
      "bool",
    );

    using fused = scaledDotProductAttention(queries, keys, values, {
      scale,
      maskMode: "array",
      maskArray: mask,
    });
    using manual = manualMaskedAttention(queries, keys, values, scale, mask);

    mxEval(fused, manual);
    expectNestedClose(fused.toList(), manual.toList());
  });

  test("gradients flow through the fused attention kernel", () => {
    const scale = Math.sqrt(1 / 3);
    using keys = array(
      [
        [
          [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
        ],
      ],
      "float32",
    );
    using values = array(
      [
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
      ],
      "float32",
    );

    const gradFn = grad((queries: MxArray) => {
      using attended = scaledDotProductAttention(queries, keys, values, {
        scale,
        maskMode: "causal",
      });
      return sum(attended);
    });

    using queries = array(
      [
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
      ],
      "float32",
    );
    using gradients = gradFn(queries);

    mxEval(gradients);
    expect(gradients.shape).toEqual([1, 1, 3, 3]);
    const flat = (gradients.toList() as number[][][][]).flat(3);
    expect(flat.some((value) => Math.abs(value) > 1e-6)).toBe(true);
  });

  test("array mask mode requires a mask array", () => {
    using queries = array([[[[1]]]], "float32");
    using keys = array([[[[1]]]], "float32");
    using values = array([[[[1]]]], "float32");

    expect(() =>
      scaledDotProductAttention(queries, keys, values, {
        scale: 1,
        maskMode: "array",
      }),
    ).toThrow("maskArray");
  });

  test("causal mode rejects explicit mask arrays", () => {
    using queries = array([[[[1]]]], "float32");
    using keys = array([[[[1]]]], "float32");
    using values = array([[[[1]]]], "float32");
    using mask = array([[1]], "bool");

    expect(() =>
      scaledDotProductAttention(queries, keys, values, {
        scale: 1,
        maskMode: "causal",
        maskArray: mask,
      }),
    ).toThrow("maskArray cannot be provided");
  });
});

describe("fast.qwenGatedDeltaUpdate", () => {
  test("computes one scalar-gated recurrent update", () => {
    if (!isMetalAvailable()) {
      return;
    }

    using q = ones([1, 1, 1, 32], "float32");
    using k = ones([1, 1, 1, 32], "float32");
    using v = array([[[[2, 3]]]], "float32");
    using g = ones([1, 1, 1], "float32");
    using beta = ones([1, 1, 1], "float32");
    using state = zeros([1, 1, 2, 32], "float32");

    const result = qwenGatedDeltaUpdate(q, k, v, g, beta, state);
    try {
      result.output.eval();
      result.state.eval();
      expect(result.output.shape).toEqual([1, 1, 1, 2]);
      expect(result.state.shape).toEqual([1, 1, 2, 32]);
      expect(result.output.toList()).toEqual([[[[64, 96]]]]);
    } finally {
      result.output.free();
      result.state.free();
    }
  });

  test("validates Qwen gated-delta input ranks and shapes", () => {
    using q = ones([1, 1, 1, 32], "float32");
    using k = ones([1, 1, 1, 32], "float32");
    using v = ones([1, 1, 1, 2], "float32");
    using g = ones([1, 1, 1], "float32");
    using beta = ones([1, 1, 1], "float32");
    using state = zeros([1, 1, 2, 32], "float32");
    using rank3Q = ones([1, 1, 32], "float32");
    using narrowQ = ones([1, 1, 1, 16], "float32");
    using narrowK = ones([1, 1, 1, 16], "float32");
    using narrowState = zeros([1, 1, 2, 16], "float32");
    using twoKeyHeadQ = ones([1, 1, 2, 32], "float32");
    using twoKeyHeadK = ones([1, 1, 2, 32], "float32");
    using moreValueHeads = ones([1, 1, 3, 2], "float32");
    using moreValueG = ones([1, 1, 3], "float32");
    using moreValueBeta = ones([1, 1, 3], "float32");
    using moreValueState = zeros([1, 3, 2, 32], "float32");
    using longerK = ones([1, 2, 1, 32], "float32");

    expect(() => qwenGatedDeltaUpdate(rank3Q, k, v, g, beta, state)).toThrow("expected q rank 4");
    expect(() => qwenGatedDeltaUpdate(narrowQ, narrowK, v, g, beta, narrowState)).toThrow(
      "positive multiple of 32",
    );
    expect(() =>
      qwenGatedDeltaUpdate(
        twoKeyHeadQ,
        twoKeyHeadK,
        moreValueHeads,
        moreValueG,
        moreValueBeta,
        moreValueState,
      ),
    ).toThrow("must be divisible");
    expect(() => qwenGatedDeltaUpdate(q, longerK, v, g, beta, state)).toThrow(
      "expected k sequence",
    );
  });
});
