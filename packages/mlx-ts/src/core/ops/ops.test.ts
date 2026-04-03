import { describe, expect, test } from "bun:test";

import { arange, array, MxArray, ones, zeros } from "../array";
import {
  abs,
  add,
  argmax,
  concatenate,
  divide,
  equal,
  exp,
  expandDims,
  greater,
  log,
  matmul,
  max,
  mean,
  min,
  multiply,
  negative,
  reshape,
  softmax,
  sqrt,
  squeeze,
  stack,
  subtract,
  sum,
  transpose,
  where,
} from "./index";

describe("Arithmetic ops", () => {
  test("add", () => {
    const a = ones([3]);
    const b = ones([3]);
    const c = add(a, b);
    c.eval();
    expect(c.toList()).toEqual([2, 2, 2]);
    a.free();
    b.free();
    c.free();
  });

  test("subtract", () => {
    const a = array([5, 3, 1]);
    const b = array([1, 1, 1]);
    const c = subtract(a, b);
    c.eval();
    expect(c.toList()).toEqual([4, 2, 0]);
    a.free();
    b.free();
    c.free();
  });

  test("multiply", () => {
    const a = array([2, 3, 4]);
    const b = array([10, 10, 10]);
    const c = multiply(a, b);
    c.eval();
    expect(c.toList()).toEqual([20, 30, 40]);
    a.free();
    b.free();
    c.free();
  });

  test("divide", () => {
    const a = array([10, 20, 30]);
    const b = array([2, 4, 5]);
    const c = divide(a, b);
    c.eval();
    expect(c.toList()).toEqual([5, 5, 6]);
    a.free();
    b.free();
    c.free();
  });

  test("negative", () => {
    const a = ones([3]);
    const b = negative(a);
    b.eval();
    expect(b.toList()).toEqual([-1, -1, -1]);
    a.free();
    b.free();
  });

  test("sqrt", () => {
    const a = array([4, 9, 16]);
    const b = sqrt(a);
    b.eval();
    expect(b.toList()).toEqual([2, 3, 4]);
    a.free();
    b.free();
  });

  test("exp of zeros is ones", () => {
    const a = zeros([3]);
    const b = exp(a);
    b.eval();
    expect(b.toList()).toEqual([1, 1, 1]);
    a.free();
    b.free();
  });

  test("log of ones is zeros", () => {
    const a = ones([3]);
    const b = log(a);
    b.eval();
    expect(b.toList()).toEqual([0, 0, 0]);
    a.free();
    b.free();
  });

  test("abs of negative values", () => {
    const a = array([-3, -1, 2]);
    const b = abs(a);
    b.eval();
    expect(b.toList()).toEqual([3, 1, 2]);
    a.free();
    b.free();
  });

  test("broadcasting: add [3,1] + [1,3]", () => {
    const a = MxArray.fromData([[1], [2], [3]]);
    const b = MxArray.fromData([[10, 20, 30]]);
    const c = add(a, b);
    expect(c.shape).toEqual([3, 3]);
    c.eval();
    expect(c.toList()).toEqual([
      [11, 21, 31],
      [12, 22, 32],
      [13, 23, 33],
    ]);
    a.free();
    b.free();
    c.free();
  });
});

describe("Linalg ops", () => {
  test("matmul of ones produces correct values", () => {
    const a = ones([3, 3]);
    const b = matmul(a, a);
    b.eval();
    expect(b.toList()).toEqual([
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
    ]);
    a.free();
    b.free();
  });

  test("matmul known values", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const c = matmul(a, b);
    c.eval();
    expect(c.toList()).toEqual([
      [19, 22],
      [43, 50],
    ]);
    a.free();
    b.free();
    c.free();
  });

  test("matmul shape [2,3] x [3,4] = [2,4]", () => {
    const a = ones([2, 3]);
    const b = ones([3, 4]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 4]);
    a.free();
    b.free();
    c.free();
  });
});

describe("Reduction ops", () => {
  test("sum all", () => {
    const a = ones([3, 3]);
    const b = sum(a);
    b.eval();
    expect(b.item()).toBe(9);
    a.free();
    b.free();
  });

  test("sum along axis 0", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = sum(a, 0);
    b.eval();
    expect(b.toList()).toEqual([5, 7, 9]);
    a.free();
    b.free();
  });

  test("mean all", () => {
    const a = array([2, 4, 6]);
    const b = mean(a);
    b.eval();
    expect(b.item()).toBeCloseTo(4);
    a.free();
    b.free();
  });

  test("max and min", () => {
    const a = array([3, 1, 4, 1, 5]);
    const mx = max(a);
    const mn = min(a);
    mx.eval();
    mn.eval();
    expect(mx.item()).toBe(5);
    expect(mn.item()).toBe(1);
    a.free();
    mx.free();
    mn.free();
  });

  test("argmax", () => {
    const a = array([3, 1, 4, 1, 5]);
    const idx = argmax(a);
    idx.eval();
    expect(idx.item()).toBe(4); // index of 5
    a.free();
    idx.free();
  });

  test("keepdims preserves rank", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = sum(a, 0, true);
    expect(b.shape).toEqual([1, 2]);
    a.free();
    b.free();
  });

  test("softmax", () => {
    const a = array([1, 2, 3]);
    const b = softmax(a, 0);
    b.eval();
    const list = b.toList() as number[];
    // Softmax should sum to 1
    const total = list.reduce((acc, v) => acc + v, 0);
    expect(total).toBeCloseTo(1.0);
    // Values should be ordered: s(1) < s(2) < s(3)
    const [first, second, third] = list;
    if (first === undefined || second === undefined || third === undefined) {
      throw new Error("Expected softmax result to contain three values");
    }
    expect(first).toBeLessThan(second);
    expect(second).toBeLessThan(third);
    a.free();
    b.free();
  });
});

describe("Shape ops", () => {
  test("reshape", () => {
    const a = arange(0, 6, 1);
    const b = reshape(a, [2, 3]);
    expect(b.shape).toEqual([2, 3]);
    b.eval();
    expect(b.toList()).toEqual([
      [0, 1, 2],
      [3, 4, 5],
    ]);
    a.free();
    b.free();
  });

  test("transpose reverses dimensions", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.shape).toEqual([2, 3]);
    const b = transpose(a);
    expect(b.shape).toEqual([3, 2]);
    b.eval();
    expect(b.toList()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    a.free();
    b.free();
  });

  test("squeeze removes size-1 dims", () => {
    const a = ones([1, 3, 1]);
    const b = squeeze(a);
    expect(b.shape).toEqual([3]);
    a.free();
    b.free();
  });

  test("expandDims adds dimension", () => {
    const a = ones([3]);
    const b = expandDims(a, 0);
    expect(b.shape).toEqual([1, 3]);
    const c = expandDims(a, 1);
    expect(c.shape).toEqual([3, 1]);
    a.free();
    b.free();
    c.free();
  });

  test("concatenate", () => {
    const a = array([1, 2]);
    const b = array([3, 4]);
    const c = concatenate([a, b]);
    c.eval();
    expect(c.toList()).toEqual([1, 2, 3, 4]);
    a.free();
    b.free();
    c.free();
  });

  test("stack", () => {
    const a = array([1, 2]);
    const b = array([3, 4]);
    const c = stack([a, b]);
    expect(c.shape).toEqual([2, 2]);
    c.eval();
    expect(c.toList()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    a.free();
    b.free();
    c.free();
  });
});

describe("Comparison ops", () => {
  test("equal", () => {
    const a = array([1, 2, 3]);
    const b = array([1, 0, 3]);
    const c = equal(a, b);
    c.eval();
    // MLX booleans are 0/1
    expect(c.toList()).toEqual([1, 0, 1]);
    a.free();
    b.free();
    c.free();
  });

  test("greater", () => {
    const a = array([3, 1, 2]);
    const b = array([2, 2, 2]);
    const c = greater(a, b);
    c.eval();
    expect(c.toList()).toEqual([1, 0, 0]);
    a.free();
    b.free();
    c.free();
  });

  test("where", () => {
    const cond = array([1, 0, 1]);
    const x = array([10, 20, 30]);
    const y = array([100, 200, 300]);
    // Need boolean condition - cast to bool
    const condBool = cond.asType("bool");
    const result = where(condBool, x, y);
    result.eval();
    expect(result.toList()).toEqual([10, 200, 30]);
    cond.free();
    condBool.free();
    x.free();
    y.free();
    result.free();
  });
});
