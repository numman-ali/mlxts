import { describe, expect, test } from "bun:test";

import { arange, array, MxArray, ones, zeros } from "../array";
import { mxEval } from "../transforms";
import {
  abs,
  add,
  argmax,
  argmin,
  asType,
  broadcastTo,
  concatenate,
  divide,
  equal,
  erf,
  exp,
  expandDims,
  flatten,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logsumexp,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  multiply,
  negative,
  notEqual,
  power,
  reciprocal,
  reshape,
  sigmoid,
  softmax,
  split,
  sqrt,
  square,
  squeeze,
  stack,
  subtract,
  sum,
  takeAlongAxis,
  tanh,
  transpose,
  tril,
  triu,
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

  test("power raises each element", () => {
    const a = array([2, 3]);
    const b = array([3, 2]);
    const c = power(a, b);
    c.eval();
    expect(c.toList()).toEqual([8, 9]);
    a.free();
    b.free();
    c.free();
  });

  test("maximum and minimum choose element-wise extrema", () => {
    const a = array([1, 5, -1]);
    const b = array([2, 4, 0]);
    const hi = maximum(a, b);
    const lo = minimum(a, b);
    hi.eval();
    lo.eval();
    expect(hi.toList()).toEqual([2, 5, 0]);
    expect(lo.toList()).toEqual([1, 4, -1]);
    a.free();
    b.free();
    hi.free();
    lo.free();
  });

  test("square multiplies values by themselves", () => {
    const a = array([-2, 3]);
    const b = square(a);
    b.eval();
    expect(b.toList()).toEqual([4, 9]);
    a.free();
    b.free();
  });

  test("sigmoid maps zero to one half", () => {
    const a = array([0]);
    const b = sigmoid(a);
    b.eval();
    expect(b.item()).toBeCloseTo(0.5, 5);
    a.free();
    b.free();
  });

  test("erf maps zero to zero", () => {
    const a = array([0]);
    const b = erf(a);
    b.eval();
    expect(b.item()).toBeCloseTo(0, 5);
    a.free();
    b.free();
  });

  test("reciprocal returns one over each value", () => {
    const a = array([2, 4], "float32");
    const b = reciprocal(a);
    b.eval();
    const list = b.toList() as number[];
    expect(list[0]).toBeCloseTo(0.5, 5);
    expect(list[1]).toBeCloseTo(0.25, 5);
    a.free();
    b.free();
  });

  test("tanh matches expected values", () => {
    const a = array([0, 1], "float32");
    const b = tanh(a);
    b.eval();
    const list = b.toList() as number[];
    expect(list[0]).toBeCloseTo(0, 5);
    expect(list[1]).toBeCloseTo(Math.tanh(1), 5);
    a.free();
    b.free();
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

  test("max and min along an axis", () => {
    const a = array([
      [1, 5, 2],
      [3, 4, 0],
    ]);
    const maxByRow = max(a, 1);
    const minByColumn = min(a, 0);
    maxByRow.eval();
    minByColumn.eval();
    expect(maxByRow.toList()).toEqual([5, 4]);
    expect(minByColumn.toList()).toEqual([1, 4, 0]);
    a.free();
    maxByRow.free();
    minByColumn.free();
  });

  test("argmax", () => {
    const a = array([3, 1, 4, 1, 5]);
    const idx = argmax(a);
    idx.eval();
    expect(idx.item()).toBe(4); // index of 5
    a.free();
    idx.free();
  });

  test("argmin", () => {
    const a = array([3, 1, 4, 1, 5]);
    const idx = argmin(a);
    idx.eval();
    expect(idx.item()).toBe(1);
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

  test("sum and mean across multiple axes", () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const summed = sum(a, [0, 2]);
    const averaged = mean(a, [0, 1]);
    summed.eval();
    averaged.eval();
    expect(summed.toList()).toEqual([14, 22]);
    expect(averaged.toList()).toEqual([4, 5]);
    a.free();
    summed.free();
    averaged.free();
  });

  test("logsumexp stays numerically stable", () => {
    const a = array([0, 0], "float32");
    const b = logsumexp(a);
    b.eval();
    expect(b.item()).toBeCloseTo(Math.log(2), 5);
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

  test("transpose with axes permutes dimensions explicitly", () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
    ]);
    const b = transpose(a, [1, 2, 0]);
    expect(b.shape).toEqual([2, 2, 1]);
    b.eval();
    expect(b.toList()).toEqual([
      [[1], [2]],
      [[3], [4]],
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

  test("squeeze removes a specific axis", () => {
    const a = ones([1, 3]);
    const b = squeeze(a, 0);
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

  test("broadcastTo expands singleton dimensions", () => {
    const a = array([[1], [2], [3]]);
    const b = broadcastTo(a, [3, 2]);
    b.eval();
    expect(b.toList()).toEqual([
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    a.free();
    b.free();
  });

  test("astype changes the visible dtype", () => {
    const a = array([1, 2, 3], "int32");
    const b = asType(a, "float32");
    expect(b.dtype).toBe("float32");
    a.free();
    b.free();
  });

  test("flatten merges a range of axes", () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const b = flatten(a, 1, 2);
    expect(b.shape).toEqual([2, 4]);
    b.eval();
    expect(b.toList()).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    a.free();
    b.free();
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

  test("takeAlongAxis gathers by index", () => {
    const a = array([
      [10, 11, 12],
      [20, 21, 22],
    ]);
    const indices = array(
      [
        [2, 1],
        [0, 2],
      ],
      "int32",
    );
    const gathered = takeAlongAxis(a, indices, 1);
    gathered.eval();
    expect(gathered.toList()).toEqual([
      [12, 11],
      [20, 22],
    ]);
    a.free();
    indices.free();
    gathered.free();
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

  test("notEqual", () => {
    const a = array([1, 2, 3]);
    const b = array([1, 0, 3]);
    const c = notEqual(a, b);
    c.eval();
    expect(c.toList()).toEqual([0, 1, 0]);
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

  test("greaterEqual, less, and lessEqual", () => {
    const a = array([1, 2, 3]);
    const b = array([1, 3, 2]);
    const ge = greaterEqual(a, b);
    const lt = less(a, b);
    const le = lessEqual(a, b);
    ge.eval();
    lt.eval();
    le.eval();
    expect(ge.toList()).toEqual([1, 0, 1]);
    expect(lt.toList()).toEqual([0, 1, 0]);
    expect(le.toList()).toEqual([1, 1, 0]);
    a.free();
    b.free();
    ge.free();
    lt.free();
    le.free();
  });

  test("comparison ops accept scalar operands symmetrically", () => {
    const values = array([1, 2, 3], "int32");
    const eq = equal(values, 2);
    const gt = greater(2, values);
    const le = lessEqual(values, 2);
    mxEval(eq, gt, le);

    expect(eq.dtype).toBe("bool");
    expect(eq.toList()).toEqual([0, 1, 0]);
    expect(gt.toList()).toEqual([1, 0, 0]);
    expect(le.toList()).toEqual([1, 1, 0]);

    values.free();
    eq.free();
    gt.free();
    le.free();
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

describe("Error paths", () => {
  test("reshape rejects incompatible element counts", () => {
    const values = arange(0, 6, 1, "int32");
    expect(() => reshape(values, [4, 4])).toThrow();
    values.free();
  });

  test("matmul rejects incompatible inner dimensions", () => {
    const left = ones([2, 3]);
    const right = ones([4, 2]);
    expect(() => matmul(left, right)).toThrow();
    left.free();
    right.free();
  });

  test("split rejects zero splits", () => {
    const values = ones([4, 4]);
    expect(() => split(values, 0)).toThrow();
    values.free();
  });

  test("squeeze rejects non-singleton axes", () => {
    const values = ones([2, 3]);
    expect(() => squeeze(values, 1)).toThrow();
    values.free();
  });

  test("takeAlongAxis rejects invalid axes", () => {
    const values = array(
      [
        [10, 11, 12],
        [20, 21, 22],
      ],
      "float32",
    );
    const indices = array([[0, 1, 2]], "int32");
    expect(() => takeAlongAxis(values, indices, 2)).toThrow();
    values.free();
    indices.free();
  });

  test("softmax accepts the precise option", () => {
    const values = array([1, 2, 3], "float32");
    const result = softmax(values, 0, { precise: true });
    result.eval();
    const total = (result.toList() as number[]).reduce((sumValue, value) => sumValue + value, 0);
    expect(total).toBeCloseTo(1, 5);
    values.free();
    result.free();
  });
});

// ---------------------------------------------------------------------------
// Scalar-array coercion
// ---------------------------------------------------------------------------

describe("scalar coercion", () => {
  test("add(array, number)", () => {
    const a = array([1, 2, 3]);
    const result = add(a, 10);
    result.eval();
    expect(result.toList()).toEqual([11, 12, 13]);
    a.free();
    result.free();
  });

  test("add(number, array)", () => {
    const a = array([1, 2, 3]);
    const result = add(5, a);
    result.eval();
    expect(result.toList()).toEqual([6, 7, 8]);
    a.free();
    result.free();
  });

  test("subtract(array, number)", () => {
    const a = array([10, 20, 30]);
    const result = subtract(a, 5);
    result.eval();
    expect(result.toList()).toEqual([5, 15, 25]);
    a.free();
    result.free();
  });

  test("multiply(array, number)", () => {
    const a = array([1, 2, 3]);
    const result = multiply(a, 0.5);
    result.eval();
    expect(result.toList()).toEqual([0.5, 1, 1.5]);
    a.free();
    result.free();
  });

  test("multiply(number, array)", () => {
    const a = array([2, 4, 6]);
    const result = multiply(2, a);
    result.eval();
    expect(result.toList()).toEqual([4, 8, 12]);
    a.free();
    result.free();
  });

  test("divide(array, number)", () => {
    const a = array([10, 20, 30]);
    const result = divide(a, 10);
    result.eval();
    expect(result.toList()).toEqual([1, 2, 3]);
    a.free();
    result.free();
  });

  test("power(array, number)", () => {
    const a = array([2, 3, 4]);
    const result = power(a, 2);
    result.eval();
    expect(result.toList()).toEqual([4, 9, 16]);
    a.free();
    result.free();
  });

  test("maximum(array, number)", () => {
    const a = array([-1, 0, 5]);
    const result = maximum(a, 0);
    result.eval();
    expect(result.toList()).toEqual([0, 0, 5]);
    a.free();
    result.free();
  });

  test("minimum(array, number)", () => {
    const a = array([1, 10, 100]);
    const result = minimum(a, 5);
    result.eval();
    expect(result.toList()).toEqual([1, 5, 5]);
    a.free();
    result.free();
  });

  test("add(number, number)", () => {
    const result = add(3, 4);
    result.eval();
    expect(result.item()).toBe(7);
    result.free();
  });
});

describe("tril / triu", () => {
  test("tril extracts lower triangle of 3x3", () => {
    const a = ones([3, 3]);
    const result = tril(a);
    result.eval();
    expect(result.toList()).toEqual([
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ]);
    a.free();
    result.free();
  });

  test("triu extracts upper triangle of 3x3", () => {
    const a = ones([3, 3]);
    const result = triu(a);
    result.eval();
    expect(result.toList()).toEqual([
      [1, 1, 1],
      [0, 1, 1],
      [0, 0, 1],
    ]);
    a.free();
    result.free();
  });

  test("tril with diagonal offset k=-1", () => {
    const a = ones([3, 3]);
    const result = tril(a, -1);
    result.eval();
    expect(result.toList()).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
    ]);
    a.free();
    result.free();
  });

  test("triu with diagonal offset k=1", () => {
    const a = ones([3, 3]);
    const result = triu(a, 1);
    result.eval();
    expect(result.toList()).toEqual([
      [0, 1, 1],
      [0, 0, 1],
      [0, 0, 0],
    ]);
    a.free();
    result.free();
  });

  test("tril on 3D input uses last two dims", () => {
    const a = ones([2, 3, 3]);
    const result = tril(a);
    result.eval();
    const expected = [
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ];
    expect(result.toList()).toEqual([expected, expected]);
    a.free();
    result.free();
  });
});

describe("split", () => {
  test("split along axis=0 into equal parts", () => {
    const a = arange(0, 6, 1, "float32");
    const r = reshape(a, [6, 1]);
    const parts = split(r, 3, 0);
    expect(parts.length).toBe(3);
    for (const p of parts) {
      p.eval();
      expect(p.shape).toEqual([2, 1]);
    }
    expect(parts[0]?.toList()).toEqual([[0], [1]]);
    expect(parts[1]?.toList()).toEqual([[2], [3]]);
    expect(parts[2]?.toList()).toEqual([[4], [5]]);
    a.free();
    r.free();
    for (const p of parts) p.free();
  });

  test("split along axis=-1 (QKV pattern)", () => {
    const a = ones([2, 12]);
    const parts = split(a, 3, -1);
    expect(parts.length).toBe(3);
    for (const p of parts) {
      p.eval();
      expect(p.shape).toEqual([2, 4]);
    }
    a.free();
    for (const p of parts) p.free();
  });
});

describe("where with scalar operands", () => {
  test("where(condition, array, number)", () => {
    const cond = array([1, 0, 1]);
    const x = array([10, 20, 30]);
    const result = where(cond, x, -1);
    result.eval();
    expect(result.toList()).toEqual([10, -1, 30]);
    cond.free();
    x.free();
    result.free();
  });

  test("where(condition, int array, number) preserves the array dtype", () => {
    const cond = array([1, 0, 1]);
    const x = array(new Int32Array([10, 20, 30]), "int32");
    const result = where(cond, x, 0);
    result.eval();
    expect(result.dtype).toBe("int32");
    expect(result.toTypedArray()).toEqual(new Int32Array([10, 0, 30]));
    cond.free();
    x.free();
    result.free();
  });

  test("where(condition, number, array)", () => {
    const cond = array([0, 1, 0]);
    const y = array([10, 20, 30]);
    const result = where(cond, 99, y);
    result.eval();
    expect(result.toList()).toEqual([10, 99, 30]);
    cond.free();
    y.free();
    result.free();
  });

  test("where(condition, number, number)", () => {
    const cond = array([1, 0]);
    const result = where(cond, 5.0, -5.0);
    result.eval();
    expect(result.toList()).toEqual([5, -5]);
    cond.free();
    result.free();
  });

  test("where with -1e9 for attention masking", () => {
    // tril produces 0/1 float — MLX treats non-zero as true in where
    const mask = tril(ones([3, 3]));
    const scores = ones([3, 3]);
    const result = where(mask, scores, -1e9);
    result.eval();
    const rows = result.toList() as number[][];
    expect(rows[0]?.[0]).toBe(1);
    expect(rows[0]?.[1]).toBeCloseTo(-1e9);
    expect(rows[2]?.[2]).toBe(1);
    mask.free();
    scores.free();
    result.free();
  });
});
