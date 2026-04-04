import { describe, expect, test } from "bun:test";

import { add, array, type MxArray, matmul, mxEval, ones, random, VERSION, zeros } from "./index";

describe("@mlxts/core", () => {
  test("exports a version string", () => {
    expect(VERSION).toBe("0.0.1");
  });

  test("PLAN.md exit criteria: ones → matmul → eval → toList", () => {
    // This is the exact deliverable from PLAN.md Phase 1:
    // "Can run: const a = mx.ones([3, 3]); const b = mx.matmul(a, a);
    //  mx.eval(b); console.log(b.tolist())"
    const a = ones([3, 3]);
    const b = matmul(a, a);
    mxEval(b);
    expect(b.toList()).toEqual([
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
    ]);
    a.free();
    b.free();
  });

  test("end-to-end: create, compute, read back", () => {
    const x = array([
      [1, 2],
      [3, 4],
    ]);
    const y = array([
      [5, 6],
      [7, 8],
    ]);
    const z = add(matmul(x, y), ones([2, 2]));
    mxEval(z);
    expect(z.toList()).toEqual([
      [20, 23],
      [44, 51],
    ]);
    x.free();
    y.free();
    z.free();
  });

  test("random module is accessible", () => {
    random.seed(42);
    const a = random.normal([3, 3]);
    expect(a.shape).toEqual([3, 3]);
    expect(a.dtype).toBe("float32");
    a.free();
  });

  test("using for automatic disposal", () => {
    let arr: MxArray;
    {
      using a = zeros([1000]);
      arr = a;
      expect(a.isDisposed).toBe(false);
    }
    expect(arr.isDisposed).toBe(true);
  });

  test("GPU execution (default stream is GPU)", () => {
    // Operations default to GPU on Apple Silicon.
    // Verify that a non-trivial computation completes correctly.
    const a = ones([64, 64]);
    const b = matmul(a, a);
    mxEval(b);
    const rows = b.toList() as number[][];
    const firstRow = rows[0];
    if (firstRow === undefined) {
      throw new Error("Expected matmul result to contain a first row");
    }
    const topLeft = firstRow[0];
    expect(topLeft).toBe(64); // dot product of 64 ones
    a.free();
    b.free();
  });
});
