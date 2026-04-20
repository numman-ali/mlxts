import { describe, expect, test } from "bun:test";

import type { MxArray } from "./array";
import { array, ones } from "./array";
import { checkStatus } from "./error";
import { extractSingleArray, ffi, OutSlot, unwrapPointer, withClosure } from "./ffi";
import { add, log, multiply, subtract } from "./ops/arithmetic";
import { matmul } from "./ops/linalg";
import { mean, softmax, sum } from "./ops/reduction";
import { stopGradient } from "./ops/shape";
import {
  checkpoint,
  clearCompileCache,
  compile,
  compileMany,
  disableCompile,
  enableCompile,
  grad,
  mxEval,
  setCompileMode,
  valueAndGrad,
} from "./transforms";

function requireCompiledOutput(outputs: MxArray[], index: number): MxArray {
  const output = outputs[index];
  if (output === undefined) {
    throw new Error(`Expected compiled output at index ${index}.`);
  }
  return output;
}

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

  test("compile returns a callable single-output transform", () => {
    const compiled = compile((x: MxArray) => add(x, 1));

    const x = array([1, 2, 3], "float32");
    const y = compiled(x);
    mxEval(y);

    expect(y.toList()).toEqual([2, 3, 4]);

    x.free();
    y.free();
  });

  test("compile transforms can be explicitly disposed", () => {
    using compiled = compile((x: MxArray) => add(x, 1));
    using x = array([1], "float32");
    using y = compiled(x);
    mxEval(y);
    expect(y.item()).toBe(2);

    compiled[Symbol.dispose]();
    expect(() => compiled(x)).toThrow("disposed");
  });

  test("checkpointed functions still differentiate correctly", () => {
    const checkpointed = checkpoint((x: MxArray) => sum(multiply(x, x)));
    const gradFn = grad((x: MxArray) => checkpointed(x));

    const x = array([3.0], "float32");
    const g = gradFn(x);
    mxEval(g);

    expect(g.toList()).toEqual([6]);

    x.free();
    g.free();
  });

  test("checkpoint transforms can be explicitly disposed", () => {
    using checkpointed = checkpoint((x: MxArray) => sum(multiply(x, x)));
    using x = array([2], "float32");
    using y = checkpointed(x);
    mxEval(y);
    expect(y.item()).toBe(4);

    checkpointed[Symbol.dispose]();
    expect(() => checkpointed(x)).toThrow("disposed");
  });

  test("compile controls are callable", () => {
    clearCompileCache();
    setCompileMode("no_fuse");
    disableCompile();
    enableCompile();
    setCompileMode("enabled");
  });

  test("compileMany returns a single output correctly", () => {
    using compiled = compileMany((x: MxArray) => {
      return [add(x, 1)];
    });

    using x = array([1, 2, 3], "float32");
    const results = compiled(x);
    using r = requireCompiledOutput(results, 0);
    mxEval(r);

    expect(r.toList()).toEqual([2, 3, 4]);
  });

  test("compileMany returns multiple outputs", () => {
    using compiled = compileMany((x: MxArray) => {
      const doubled = add(x, x);
      const tripled = add(doubled, x);
      return [doubled, tripled];
    });

    using x = array([1, 2, 3], "float32");
    const results = compiled(x);
    using d = requireCompiledOutput(results, 0);
    using t = requireCompiledOutput(results, 1);
    mxEval(d, t);

    expect(d.toList()).toEqual([2, 4, 6]);
    expect(t.toList()).toEqual([3, 6, 9]);
  });

  test("compileMany with 3 outputs returns correct shapes and values", () => {
    using compiled = compileMany((a: MxArray, b: MxArray) => {
      const s = add(a, b);
      const d = subtract(a, b);
      const p = multiply(a, b);
      return [s, d, p];
    });

    using a = array([4, 5], "float32");
    using b = array([1, 2], "float32");
    const results = compiled(a, b);
    using sum_ = requireCompiledOutput(results, 0);
    using diff = requireCompiledOutput(results, 1);
    using prod = requireCompiledOutput(results, 2);
    mxEval(sum_, diff, prod);

    expect(sum_.toList()).toEqual([5, 7]);
    expect(diff.toList()).toEqual([3, 3]);
    expect(prod.toList()).toEqual([4, 10]);
  });

  test("compileMany disposal frees native resources", () => {
    using compiled = compileMany((x: MxArray) => [add(x, 1), multiply(x, 2)]);
    using x = array([1], "float32");
    const results = compiled(x);
    using a_ = requireCompiledOutput(results, 0);
    using b_ = requireCompiledOutput(results, 1);
    mxEval(a_, b_);
    expect(a_.item()).toBe(2);
    expect(b_.item()).toBe(2);

    compiled[Symbol.dispose]();
    expect(() => compiled(x)).toThrow("disposed");
  });
});

describe("closure bridge", () => {
  test("identity closure round-trip returns input unchanged", () => {
    const identityFn = (x: MxArray) => [x];

    withClosure(identityFn, (closurePtr) => {
      const input = array([1, 2, 3]);

      // Build input vector
      const inputVec = unwrapPointer(ffi.mlx_vector_array_new(), "test_input_vec");
      try {
        checkStatus(ffi.mlx_vector_array_append_value(inputVec, input._ctx), "test_append");

        // Apply the closure
        const outSlot = new OutSlot();
        checkStatus(
          ffi.mlx_closure_apply(outSlot.prepare(), closurePtr, inputVec),
          "closure_apply",
        );
        const resultVec = outSlot.read("closure apply result");

        try {
          const result = extractSingleArray(resultVec, "identity");
          mxEval(result);
          expect(result.toList()).toEqual([1, 2, 3]);
          result.free();
        } finally {
          ffi.mlx_vector_array_free(resultVec);
        }
      } finally {
        ffi.mlx_vector_array_free(inputVec);
      }
      input.free();
    });
  });
});

describe("valueAndGrad", () => {
  test("gradient of x^2", () => {
    // f(x) = sum(x*x) for scalar-like input [3.0]
    // f'(x) = 2x = [6.0]
    const fn = (x: MxArray) => sum(multiply(x, x));
    const vgFn = valueAndGrad(fn);

    const x = array([3.0]);
    const [value, gradient] = vgFn(x);
    mxEval(value, gradient);

    expect(value.item()).toBeCloseTo(9.0, 5);
    expect(gradient.toList()).toEqual([6]);

    x.free();
    value.free();
    gradient.free();
  });

  test("gradient of linear function 2x + 1", () => {
    const two = array([2.0]);
    const one = array([1.0]);
    const fn = (x: MxArray) => sum(add(multiply(two, x), one));
    const gradFn = grad(fn);

    const x = array([5.0]);
    const g = gradFn(x);
    mxEval(g);

    expect(g.toList()).toEqual([2]);

    x.free();
    g.free();
    two.free();
    one.free();
  });

  test("gradient of polynomial x^3 - 2x^2 + x", () => {
    // f(x) = x^3 - 2x^2 + x
    // f'(x) = 3x^2 - 4x + 1
    // f'(2) = 12 - 8 + 1 = 5
    const two = array([2.0]);
    const fn = (x: MxArray) => {
      const x2 = multiply(x, x);
      const x3 = multiply(x2, x);
      const term2 = multiply(two, x2);
      return sum(add(subtract(x3, term2), x));
    };
    const gradFn = grad(fn);

    const x = array([2.0]);
    const g = gradFn(x);
    mxEval(g);

    expect((g.toList() as number[])[0]).toBeCloseTo(5.0, 4);

    x.free();
    g.free();
    two.free();
  });

  test("gradient through sum reduction", () => {
    // f(x) = sum(x * x), f'(x_i) = 2 * x_i
    const fn = (x: MxArray) => sum(multiply(x, x));
    const gradFn = grad(fn);

    const x = array([1, 2, 3], "float32");
    const g = gradFn(x);
    mxEval(g);

    expect(g.toList()).toEqual([2, 4, 6]);

    x.free();
    g.free();
  });

  test("gradient through mean reduction", () => {
    // f(x) = mean(x * x), f'(x_i) = 2 * x_i / n
    const fn = (x: MxArray) => mean(multiply(x, x));
    const gradFn = grad(fn);

    const x = array([1, 2, 3], "float32");
    const g = gradFn(x);
    mxEval(g);

    const gList = g.toList() as number[];
    expect(gList[0]).toBeCloseTo(2 / 3, 4);
    expect(gList[1]).toBeCloseTo(4 / 3, 4);
    expect(gList[2]).toBeCloseTo(6 / 3, 4);

    x.free();
    g.free();
  });

  test("grad transforms can be explicitly disposed", () => {
    using transform = grad((x: MxArray) => sum(multiply(x, x)));
    using x = array([3], "float32");
    using gradient = transform(x);
    mxEval(gradient);
    expect(gradient.toList()).toEqual([6]);

    transform[Symbol.dispose]();
    expect(() => transform(x)).toThrow("disposed");
  });

  test("gradient through matmul", () => {
    // f(W) = sum(W @ x) where x is fixed
    // d/dW sum(W @ x) = ones @ x^T (each row of grad = x^T)
    const x = array([[1], [2]], "float32"); // 2x1
    const fn = (W: MxArray) => sum(matmul(W, x));
    const gradFn = grad(fn);

    const W = array(
      [
        [1, 0],
        [0, 1],
      ],
      "float32",
    ); // 2x2
    const g = gradFn(W);
    mxEval(g);

    // Each row of the gradient should be x^T = [1, 2]
    expect(g.toList()).toEqual([
      [1, 2],
      [1, 2],
    ]);
    expect(g.shape).toEqual([2, 2]);

    x.free();
    W.free();
    g.free();
  });

  test("valueAndGrad returns tuple [value, gradient]", () => {
    const fn = (x: MxArray) => sum(multiply(x, x));
    const vgFn = valueAndGrad(fn);

    const x = array([4.0]);
    const result = vgFn(x);

    // Should be a tuple (array of length 2)
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);

    const [value, gradient] = result;
    mxEval(value, gradient);
    expect(value.item()).toBeCloseTo(16.0, 5);
    expect(gradient.toList()).toEqual([8]);

    x.free();
    value.free();
    gradient.free();
  });

  test("valueAndGrad transforms can be explicitly disposed", () => {
    using transform = valueAndGrad((x: MxArray) => sum(multiply(x, x)));
    using x = array([4], "float32");
    const [value, gradient] = transform(x);
    try {
      mxEval(value, gradient);
      expect(value.item()).toBeCloseTo(16, 5);
      expect(gradient.toList()).toEqual([8]);
    } finally {
      value.free();
      gradient.free();
    }

    transform[Symbol.dispose]();
    expect(() => transform(x)).toThrow("disposed");
  });

  test("single argnum returns bare MxArray gradient", () => {
    const fn = (x: MxArray) => sum(multiply(x, x));
    const gradFn = grad(fn); // default argnums=0

    const x = array([5.0]);
    const g = gradFn(x);

    // g should be a single MxArray, not an array
    expect(g).toHaveProperty("_ctx");
    expect(g).toHaveProperty("shape");

    mxEval(g);
    expect(g.toList()).toEqual([10]);

    x.free();
    g.free();
  });

  test("multiple argnums returns array of gradients", () => {
    // f(a, b) = sum(a * b)
    const fn = (a: MxArray, b: MxArray) => sum(multiply(a, b));
    const vgFn = valueAndGrad(fn, [0, 1]);

    const a = array([1, 2, 3], "float32");
    const b = array([4, 5, 6], "float32");
    const [value, grads] = vgFn(a, b);

    // grads should be an array
    expect(Array.isArray(grads)).toBe(true);
    expect(grads.length).toBe(2);

    // Narrow types via runtime check (noUncheckedIndexedAccess)
    if (grads.length < 2) throw new Error("expected 2 gradients");
    const g0 = grads[0] as import("./array").MxArray;
    const g1 = grads[1] as import("./array").MxArray;
    mxEval(value, g0, g1);

    // sum(a * b) = 1*4 + 2*5 + 3*6 = 32
    expect(value.item()).toBeCloseTo(32.0, 5);
    // d/da sum(a*b) = b
    expect(g0.toList()).toEqual([4, 5, 6]);
    // d/db sum(a*b) = a
    expect(g1.toList()).toEqual([1, 2, 3]);

    a.free();
    b.free();
    value.free();
    g0.free();
    g1.free();
  });

  test("stopGradient blocks gradient flow", () => {
    // f(x) = sum(x * stopGradient(x))
    // d/dx = stopGradient(x) = x (the stopped part is treated as a constant)
    const fn = (x: MxArray) => sum(multiply(x, stopGradient(x)));
    const gradFn = grad(fn);

    const x = array([3.0]);
    const g = gradFn(x);
    mxEval(g);

    expect((g.toList() as number[])[0]).toBeCloseTo(3.0, 5); // not 6.0

    x.free();
    g.free();
  });

  test("error in loss function is propagated as original JS exception", () => {
    const fn = (_x: MxArray): MxArray => {
      throw new Error("intentional test error");
    };
    const gradFn = grad(fn);

    const x = array([1.0]);
    expect(() => gradFn(x)).toThrow("intentional test error");
    x.free();
  });

  test("non-scalar loss throws a clear scalar-loss error", () => {
    const gradFn = grad((x: MxArray) => x);

    const x = array([1, 2, 3], "float32");
    expect(() => gradFn(x)).toThrow("expected fn to return a scalar MxArray");
    x.free();
  });

  test("numerical gradient check for x^2", () => {
    // float32 on GPU — use larger eps to avoid float32 precision issues
    const fn = (x: MxArray) => sum(multiply(x, x));
    const gradFn = grad(fn);

    const xVal = 2.5;
    const eps = 1e-3;

    // Autograd
    const x = array([xVal], "float32");
    const g = gradFn(x);
    mxEval(g);
    const gList = g.toList() as number[];
    const autoGrad = gList[0] ?? 0;

    // Numerical: (f(x+eps) - f(x-eps)) / (2*eps)
    const xPlus = array([xVal + eps], "float32");
    const xMinus = array([xVal - eps], "float32");
    const fPlus = fn(xPlus);
    const fMinus = fn(xMinus);
    mxEval(fPlus, fMinus);
    const numGrad = (fPlus.item() - fMinus.item()) / (2 * eps);

    // With eps=1e-3 and float32, expect ~2 decimal places of agreement
    expect(autoGrad).toBeCloseTo(numGrad, 2);

    x.free();
    g.free();
    xPlus.free();
    xMinus.free();
    fPlus.free();
    fMinus.free();
  });

  test("higher-order gradient: second derivative of x^3", () => {
    // f(x) = sum(x^3)
    // f'(x) = 3x^2
    // f''(x) = 6x
    // f''(2) = 12
    //
    // grad(fn) returns an array matching input shape. The outer grad needs
    // a scalar return, so we wrap with sum.
    const fn = (x: MxArray) => sum(multiply(multiply(x, x), x));
    const gradFn = grad(fn);
    const grad2Fn = grad((x: MxArray) => sum(gradFn(x)));

    const x = array([2.0]);
    const g2 = grad2Fn(x);
    mxEval(g2);

    expect((g2.toList() as number[])[0]).toBeCloseTo(12.0, 4);

    x.free();
    g2.free();
  });

  test("composed training-style loss differentiates through softmax and log", () => {
    const loss = (x: MxArray) => {
      const probs = softmax(x);
      const logProbs = log(probs);
      return sum(multiply(probs, logProbs));
    };
    const gradFn = grad(loss);

    const x = array([0.5, -0.25], "float32");
    const gradValue = gradFn(x);
    mxEval(gradValue);
    const gradList = gradValue.toList() as number[];

    const eps = 1e-3;
    const numerical: number[] = [];
    for (let index = 0; index < 2; index++) {
      const plus = index === 0 ? [0.5 + eps, -0.25] : [0.5, -0.25 + eps];
      const minus = index === 0 ? [0.5 - eps, -0.25] : [0.5, -0.25 - eps];
      const xPlus = array(plus, "float32");
      const xMinus = array(minus, "float32");
      const fPlus = loss(xPlus);
      const fMinus = loss(xMinus);
      mxEval(fPlus, fMinus);
      numerical.push((fPlus.item() - fMinus.item()) / (2 * eps));
      xPlus.free();
      xMinus.free();
      fPlus.free();
      fMinus.free();
    }

    expect(gradList[0]).toBeCloseTo(numerical[0] ?? 0, 2);
    expect(gradList[1]).toBeCloseTo(numerical[1] ?? 0, 2);

    x.free();
    gradValue.free();
  });

  test("argnums validation: out of range throws clear error", () => {
    const fn = (x: MxArray) => sum(x);
    const gradFn = grad(fn, 5);

    const x = array([1.0]);
    expect(() => gradFn(x)).toThrow("argnum 5 is out of range");
    x.free();
  });

  test("argnums validation: negative throws clear error", () => {
    const fn = (x: MxArray) => sum(x);
    const gradFn = grad(fn, -1);

    const x = array([1.0]);
    expect(() => gradFn(x)).toThrow("Invalid argnum -1");
    x.free();
  });

  test("argnums validation: duplicates throw clear error", () => {
    const fn = (a: MxArray, _b: MxArray) => sum(multiply(a, a));
    const gradFn = grad(fn, [0, 0]);

    const a = array([1.0]);
    const b = array([2.0]);
    expect(() => gradFn(a, b)).toThrow("Duplicate argnums");
    a.free();
    b.free();
  });
});
