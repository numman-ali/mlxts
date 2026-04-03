import { describe, expect, test } from "bun:test";

import { ones } from "./array";
import { add } from "./ops/arithmetic";
import { matmul } from "./ops/linalg";
import { mxEval } from "./transforms";

describe("transforms", () => {
  test("eval forces computation of a single array", () => {
    const a = ones([3]);
    const b = ones([3]);
    const c = add(a, b);
    mxEval(c);
    expect(c.toList()).toEqual([2, 2, 2]);
    a.free();
    b.free();
    c.free();
  });

  test("eval forces multiple arrays at once", () => {
    const a = ones([3, 3]);
    const b = matmul(a, a);
    const c = add(a, a);
    mxEval(b, c);
    expect(b.toList()).toEqual([
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
    ]);
    expect(c.toList()).toEqual([
      [2, 2, 2],
      [2, 2, 2],
      [2, 2, 2],
    ]);
    a.free();
    b.free();
    c.free();
  });

  test("eval with no arguments is a no-op", () => {
    mxEval(); // should not throw
  });
});
