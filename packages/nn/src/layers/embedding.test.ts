import { describe, expect, test } from "bun:test";
import { array, MxArray, mxEval } from "@mlxts/core";
import { Embedding } from "./embedding";

describe("Embedding", () => {
  test("forward with int32 indices", () => {
    using emb = new Embedding(10, 4);
    const indices = array([0, 1, 2], "int32");
    const out = emb.forward(indices);
    mxEval(out);

    expect(out.shape).toEqual([3, 4]);

    indices.free();
    out.free();
  });

  test("forward with 2D indices", () => {
    using emb = new Embedding(10, 4);
    const indices = MxArray.fromData([0, 1, 2, 3, 4, 5], [2, 3], "int32");
    const out = emb.forward(indices);
    mxEval(out);

    expect(out.shape).toEqual([2, 3, 4]);

    indices.free();
    out.free();
  });

  test("float indices throw", () => {
    using emb = new Embedding(10, 4);
    const indices = array([0.0, 1.0, 2.0]);

    expect(() => emb.forward(indices)).toThrow("integer dtype");

    indices.free();
  });

  test("asLinear output shape", () => {
    using emb = new Embedding(10, 4);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const out = emb.asLinear(x);
    mxEval(out);

    expect(out.shape).toEqual([2, 10]);

    x.free();
    out.free();
  });

  test("asLinear validates the input feature dimension", () => {
    using emb = new Embedding(10, 4);
    const x = MxArray.fromData([1, 2, 3], [1, 3]);

    expect(() => emb.asLinear(x)).toThrow("expected input last dimension 4");

    x.free();
  });

  test("invalid constructor throws", () => {
    expect(() => new Embedding(0, 4)).toThrow();
  });

  test("parameters have weight only", () => {
    using emb = new Embedding(10, 4);
    const params = emb.parameters();

    expect(Object.keys(params)).toEqual(["weight"]);
    expect(params.weight).toBe(emb.weight);
  });

  test("refreshes the cached projection weight when the embedding matrix changes", () => {
    using emb = new Embedding(3, 2);
    const initialWeight = emb.weight;
    emb.weight = MxArray.fromData([1, 0, 0, 1, 1, 1], [3, 2]);

    const x = MxArray.fromData([1, 1], [1, 2]);
    const firstOut = emb.asLinear(x);
    mxEval(firstOut);
    expect(firstOut.toList()).toEqual([[1, 1, 2]]);
    firstOut.free();

    initialWeight.free();
    const previousWeight = emb.weight;
    emb.weight = MxArray.fromData([2, 0, 0, 3, 1, 1], [3, 2]);
    previousWeight.free();

    const secondOut = emb.asLinear(x);
    mxEval(secondOut);
    expect(secondOut.toList()).toEqual([[2, 3, 2]]);
    secondOut.free();
    x.free();
  });
});
