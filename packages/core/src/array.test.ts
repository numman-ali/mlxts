import { describe, expect, test } from "bun:test";

import { arange, array, full, MxArray, ones, zeros } from "./array";

describe("MxArray", () => {
  test("fromData with flat JS array", () => {
    const a = MxArray.fromData([1, 2, 3]);
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toBe("float32");
    expect(a.ndim).toBe(1);
    expect(a.size).toBe(3);
    a.free();
  });

  test("fromData with nested JS array", () => {
    const a = MxArray.fromData([
      [1, 2],
      [3, 4],
    ]);
    expect(a.shape).toEqual([2, 2]);
    expect(a.ndim).toBe(2);
    expect(a.size).toBe(4);
    a.free();
  });

  test("fromData with Float32Array", () => {
    const a = MxArray.fromData(new Float32Array([1.5, 2.5, 3.5]));
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toBe("float32");
    a.free();
  });

  test("fromData with explicit shape", () => {
    const a = MxArray.fromData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.ndim).toBe(2);
    a.free();
  });

  test("fromData honors an explicit dtype instead of reinterpreting the buffer", () => {
    const a = MxArray.fromData([1.9, -2.2, 3.1], undefined, "int32");
    expect(a.dtype).toBe("int32");
    expect(a.toList()).toEqual([1, -2, 3]);
    a.free();
  });

  test("fromData can create a scalar with an explicit dtype", () => {
    const a = MxArray.fromData([7], [], "float64");
    expect(a.ndim).toBe(0);
    expect(a.dtype).toBe("float64");
    expect(a.item()).toBe(7);
    a.free();
  });

  test("toList returns correct nested array", () => {
    const a = MxArray.fromData([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const list = a.toList();
    expect(list).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    a.free();
  });

  test("toList for 1D array", () => {
    const a = MxArray.fromData([10, 20, 30]);
    expect(a.toList()).toEqual([10, 20, 30]);
    a.free();
  });

  test("item extracts scalar value", () => {
    const a = MxArray.fromData(new Float32Array([42.0]));
    expect(a.item()).toBeCloseTo(42.0);
    a.free();
  });

  test("toTypedArray returns Float32Array copy", () => {
    const a = MxArray.fromData([1, 2, 3, 4]);
    const ta = a.toTypedArray();
    expect(ta).toBeInstanceOf(Float32Array);
    expect(Array.from(ta)).toEqual([1, 2, 3, 4]);
    a.free();
  });

  test("disposal sets isDisposed flag", () => {
    const a = zeros([3]);
    expect(a.isDisposed).toBe(false);
    a.free();
    expect(a.isDisposed).toBe(true);
  });

  test("double free does not crash", () => {
    const a = zeros([3]);
    a.free();
    a.free(); // should be no-op
    expect(a.isDisposed).toBe(true);
  });

  test("using declaration disposes at end of scope", () => {
    let arr: MxArray;
    {
      using a = zeros([3]);
      arr = a;
      expect(a.isDisposed).toBe(false);
    }
    expect(arr.isDisposed).toBe(true);
  });

  test("asType converts dtype", () => {
    const a = MxArray.fromData([1.5, 2.5, 3.5]);
    const b = a.asType("int32");
    expect(b.dtype).toBe("int32");
    b.eval();
    expect(b.toList()).toEqual([1, 2, 3]);
    a.free();
    b.free();
  });
});

describe("Factory functions", () => {
  test("zeros creates zero-filled array", () => {
    const a = zeros([2, 3]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.dtype).toBe("float32");
    a.eval();
    const list = a.toList() as number[][];
    expect(list).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    a.free();
  });

  test("ones creates one-filled array", () => {
    const a = ones([3]);
    a.eval();
    expect(a.toList()).toEqual([1, 1, 1]);
    a.free();
  });

  test("full creates constant-filled array", () => {
    const a = full([2, 2], 7.0);
    a.eval();
    expect(a.toList()).toEqual([
      [7, 7],
      [7, 7],
    ]);
    a.free();
  });

  test("arange creates evenly spaced values", () => {
    const a = arange(0, 5, 1);
    a.eval();
    expect(a.toList()).toEqual([0, 1, 2, 3, 4]);
    a.free();
  });

  test("arange with fractional step", () => {
    const a = arange(0, 1, 0.25);
    a.eval();
    const list = a.toList() as number[];
    expect(list.length).toBe(4);
    expect(list[0]).toBeCloseTo(0);
    expect(list[1]).toBeCloseTo(0.25);
    expect(list[2]).toBeCloseTo(0.5);
    expect(list[3]).toBeCloseTo(0.75);
    a.free();
  });

  test("array convenience function", () => {
    const a = array([1, 2, 3]);
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toBe("float32");
    a.free();
  });

  test("zeros with int32 dtype", () => {
    const a = zeros([3], "int32");
    expect(a.dtype).toBe("int32");
    a.free();
  });
});

describe("scalar array()", () => {
  test("number creates 0-dim scalar", () => {
    const a = array(5.0);
    expect(a.shape).toEqual([]);
    expect(a.ndim).toBe(0);
    expect(a.dtype).toBe("float32");
    a.eval();
    expect(a.item()).toBeCloseTo(5.0, 5);
    a.free();
  });

  test("number with explicit dtype", () => {
    const a = array(7, "int32");
    expect(a.shape).toEqual([]);
    expect(a.dtype).toBe("int32");
    a.eval();
    expect(a.item()).toBe(7);
    a.free();
  });

  test("negative number creates scalar", () => {
    const a = array(-3.14);
    expect(a.shape).toEqual([]);
    a.eval();
    expect(a.item()).toBeCloseTo(-3.14, 4);
    a.free();
  });

  test("zero creates scalar", () => {
    const a = array(0);
    expect(a.shape).toEqual([]);
    a.eval();
    expect(a.item()).toBe(0);
    a.free();
  });
});
