import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { getDataPointer } from "./array-ffi-data";

describe("array-ffi-data", () => {
  test("getDataPointer returns readable pointers for every supported direct dtype", () => {
    const fixtures = [
      { tensor: array([1, 2], "float32"), dtype: "float32" as const },
      { tensor: array([1, 2], "float64"), dtype: "float64" as const },
      { tensor: array([1, 2], "int8"), dtype: "int8" as const },
      { tensor: array([1, 2], "int16"), dtype: "int16" as const },
      { tensor: array([1, 2], "int32"), dtype: "int32" as const },
      { tensor: array([1, 2], "uint8"), dtype: "uint8" as const },
      { tensor: array([1, 2], "uint16"), dtype: "uint16" as const },
      { tensor: array([1, 2], "uint32"), dtype: "uint32" as const },
      { tensor: array([0, 1], "bool"), dtype: "bool" as const },
    ];

    try {
      for (const fixture of fixtures) {
        expect(getDataPointer(fixture.tensor, fixture.dtype)).toBeTruthy();
      }
    } finally {
      for (const fixture of fixtures) {
        fixture.tensor.free();
      }
    }
  });
});
