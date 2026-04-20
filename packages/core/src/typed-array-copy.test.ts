import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { copyTypedArrayFromContiguous, scalarFromTypedArrayCopy } from "./typed-array-copy";

describe("typed-array-copy", () => {
  test("copies supported direct dtypes into the matching typed-array classes", () => {
    const cases = [
      { dtype: "float64", values: [1, 2], ctor: Float64Array },
      { dtype: "int8", values: [1, -2], ctor: Int8Array },
      { dtype: "int16", values: [1, -2], ctor: Int16Array },
      { dtype: "int32", values: [1, -2], ctor: Int32Array },
      { dtype: "bool", values: [1, 0], ctor: Uint8Array },
      { dtype: "uint16", values: [1, 2], ctor: Uint16Array },
      { dtype: "uint32", values: [1, 2], ctor: Uint32Array },
    ] as const;

    for (const { dtype, values, ctor } of cases) {
      using tensor = array(values, dtype);
      const copied = copyTypedArrayFromContiguous(tensor);
      expect(copied).toBeInstanceOf(ctor);
      expect(Array.from(copied)).toEqual(
        values.map((value) => (dtype === "bool" ? (value !== 0 ? 1 : 0) : value)),
      );
    }
  });

  test("falls back through float32 for non-direct storage dtypes", () => {
    using tensor = array([1.5, -2.25], "float16");
    const copied = copyTypedArrayFromContiguous(tensor);
    expect(copied).toBeInstanceOf(Float32Array);
    expect(Array.from(copied)).toEqual([1.5, -2.25]);
  });

  test("reads a scalar from a copied typed array", () => {
    expect(scalarFromTypedArrayCopy(new Float32Array([3.5]))).toBe(3.5);
  });
});
