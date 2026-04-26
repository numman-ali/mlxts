import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { Embedding } from "./embedding";
import { QuantizedEmbedding } from "./quantized-embedding";

function meanAbsoluteError(actual: unknown, expected: unknown): number {
  const actualValues = flattenNumbers(actual);
  const expectedValues = flattenNumbers(expected);
  expect(actualValues.length).toBe(expectedValues.length);
  let error = 0;
  for (let index = 0; index < actualValues.length; index += 1) {
    error += Math.abs((actualValues[index] ?? 0) - (expectedValues[index] ?? 0));
  }
  return error / Math.max(actualValues.length, 1);
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

function fixtureEmbedding(): Embedding {
  const embedding = new Embedding(6, 64);
  embedding.weight.free();
  embedding.weight = array(
    Array.from({ length: 6 }, (_, row) =>
      Array.from({ length: 64 }, (_, column) => (row * 64 + column - 96) / 128),
    ),
    "float32",
  );
  return embedding;
}

describe("QuantizedEmbedding", () => {
  test("forward dequantizes only selected rows and approximates dense embedding output", () => {
    using embedding = fixtureEmbedding();
    using quantized = QuantizedEmbedding.fromEmbedding(embedding, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    using indices = array(
      [
        [0, 3, 5],
        [2, 1, 4],
      ],
      "int32",
    );
    using expected = embedding.forward(indices);
    using actual = quantized.forward(indices);
    mxEval(expected, actual);

    expect(actual.shape).toEqual([2, 3, 64]);
    expect(quantized.weight.shape).toEqual([6, 8]);
    expect(quantized.scales.shape).toEqual([6, 2]);
    expect(quantized.biases?.shape).toEqual([6, 2]);
    expect(meanAbsoluteError(actual.toList(), expected.toList())).toBeLessThan(0.04);
  });

  test("asLinear uses the packed embedding as a tied output projection", () => {
    using embedding = fixtureEmbedding();
    using quantized = QuantizedEmbedding.fromEmbedding(embedding, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    using input = array([Array.from({ length: 64 }, (_, index) => (index - 16) / 64)]);
    using expected = embedding.asLinear(input);
    using actual = quantized.asLinear(input);
    mxEval(expected, actual);

    expect(actual.shape).toEqual([1, 6]);
    expect(meanAbsoluteError(actual.toList(), expected.toList())).toBeLessThan(0.25);
  });

  test("non-affine modes omit quantization biases", () => {
    using embedding = fixtureEmbedding();
    using quantized = QuantizedEmbedding.fromEmbedding(embedding, {
      bits: 4,
      groupSize: 32,
      mode: "mxfp4",
    });

    expect(quantized.biases).toBeNull();
    expect(quantized.weight.dtype).toBe("uint32");
  });

  test("constructor exposes quantization metadata and dense reconstruction", () => {
    using embedding = fixtureEmbedding();
    using quantized = QuantizedEmbedding.fromEmbedding(embedding, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    using dense = quantized.toEmbedding();
    using indices = array([[0, 4]], "int32");
    using expected = embedding.forward(indices);
    using actual = dense.forward(indices);
    mxEval(expected, actual);

    expect(quantized.numEmbeddings).toBe(6);
    expect(quantized.embeddingDims).toBe(64);
    expect(quantized.groupSize).toBe(32);
    expect(quantized.bits).toBe(4);
    expect(quantized.mode).toBe("affine");
    expect(dense.weight.shape).toEqual([6, 64]);
    expect(meanAbsoluteError(actual.toList(), expected.toList())).toBeLessThan(0.04);
  });

  test("direct construction supports non-affine default storage", () => {
    using quantized = new QuantizedEmbedding(3, 32, { mode: "mxfp4" });

    expect(quantized.numEmbeddings).toBe(3);
    expect(quantized.embeddingDims).toBe(32);
    expect(quantized.groupSize).toBe(32);
    expect(quantized.bits).toBe(4);
    expect(quantized.mode).toBe("mxfp4");
    expect(quantized.biases).toBeNull();
  });

  test("rejects non-integer indices and invalid tied projection dimensions", () => {
    using embedding = new QuantizedEmbedding(4, 32, { groupSize: 32, bits: 4 });
    using floatIndices = array([0, 1], "float32");
    using wrongHidden = array([[1, 2, 3]], "float32");

    expect(() => embedding.forward(floatIndices)).toThrow("integer dtype");
    expect(() => embedding.asLinear(wrongHidden)).toThrow("expected input last dimension 32");
  });

  test("rejects invalid constructor dimensions and packed layouts", () => {
    expect(() => new QuantizedEmbedding(0, 32)).toThrow("numEmbeddings must be > 0");
    expect(() => new QuantizedEmbedding(4, 0)).toThrow("embeddingDims must be > 0");
    expect(() => new QuantizedEmbedding(4, 40, { groupSize: 32 })).toThrow(
      "must be divisible by groupSize",
    );
    expect(() => new QuantizedEmbedding(4, 30, { groupSize: 10, bits: 5 })).toThrow(
      "valid packed layout",
    );
  });
});
