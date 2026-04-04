import { toArrayBuffer } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { defaultStream } from "./device";
import { DTYPE_TO_MLX } from "./dtype";
import { initializeErrorHandler } from "./error";
import { ffi, ptr, readI32, readPtr, unwrapPointer } from "./ffi";

// Initialize error handler before any tests
initializeErrorHandler();

describe("FFI layer", () => {
  test("dylib loads and ffi object has expected symbols", () => {
    expect(ffi.mlx_array_new).toBeDefined();
    expect(ffi.mlx_array_free).toBeDefined();
    expect(ffi.mlx_add).toBeDefined();
    expect(ffi.mlx_matmul).toBeDefined();
  });

  test("can create and free an empty array", () => {
    // mlx_array_new() returns an empty handle (ctx=NULL) → Bun returns null
    const ctx = ffi.mlx_array_new();
    expect(ctx).toBeNull(); // empty handle has null ctx

    // Free should still succeed on an empty handle
    const status = ffi.mlx_array_free(ctx);
    expect(status).toBe(0);
  });

  test("can create a scalar float array", () => {
    const ctx = ffi.mlx_array_new_float(42.0);
    expect(ctx).not.toBeNull();

    // mlx_array_ndim returns size_t, which Bun exposes as a number for small values.
    const ndim = ffi.mlx_array_ndim(ctx);
    expect(ndim).toBe(0);

    // size=1 for a scalar
    const size = ffi.mlx_array_size(ctx);
    expect(size).not.toBeNull();

    // Check dtype = float32 (10)
    const dtype = ffi.mlx_array_dtype(ctx);
    expect(dtype).toBe(DTYPE_TO_MLX.float32);

    ffi.mlx_array_free(ctx);
  });

  test("can create array from data buffer", () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const shape = new Int32Array([2, 3]);

    const ctx = ffi.mlx_array_new_data(ptr(data), ptr(shape), 2, DTYPE_TO_MLX.float32);
    expect(ctx).not.toBeNull();

    // Read shape back
    const shapePtr = unwrapPointer(ffi.mlx_array_shape(ctx), "mlx_array_shape");
    expect(readI32(shapePtr, 0)).toBe(2);
    expect(readI32(shapePtr, 4)).toBe(3);

    ffi.mlx_array_free(ctx);
  });

  test("can evaluate an array and read data", () => {
    const data = new Float32Array([1.5, 2.5, 3.5]);
    const shape = new Int32Array([3]);

    const ctx = ffi.mlx_array_new_data(ptr(data), ptr(shape), 1, DTYPE_TO_MLX.float32);

    const evalStatus = ffi.mlx_array_eval(ctx);
    expect(evalStatus).toBe(0);

    const dataPtr = unwrapPointer(ffi.mlx_array_data_float32(ctx), "mlx_array_data_float32");

    // Read the data back (zero-copy view)
    const result = new Float32Array(toArrayBuffer(dataPtr, 0, 12));
    expect(result[0]).toBeCloseTo(1.5);
    expect(result[1]).toBeCloseTo(2.5);
    expect(result[2]).toBeCloseTo(3.5);

    ffi.mlx_array_free(ctx);
  });

  test("can perform mlx_add with output pointer", () => {
    const s = defaultStream();

    const a = ffi.mlx_array_new_float(3.0);
    const b = ffi.mlx_array_new_float(4.0);

    // Output buffer: 8 bytes holding an mlx_array struct (ctx pointer)
    const outBuf = new Uint8Array(8);
    const outPtr = ptr(outBuf);

    const status = ffi.mlx_add(outPtr, a, b, s);
    expect(status).toBe(0);

    // Read the result ctx
    const resultCtx = unwrapPointer(readPtr(outPtr, 0), "mlx_add result");

    // Eval and check value
    ffi.mlx_array_eval(resultCtx);
    const itemBuf = new Float32Array(1);
    const itemStatus = ffi.mlx_array_item_float32(ptr(itemBuf), resultCtx);
    expect(itemStatus).toBe(0);
    expect(itemBuf[0]).toBe(7.0);

    ffi.mlx_array_free(a);
    ffi.mlx_array_free(b);
    ffi.mlx_array_free(resultCtx);
  });

  test("can create zeros and ones arrays", () => {
    const s = defaultStream();
    const shape = new Int32Array([2, 3]);
    const outBuf = new Uint8Array(8);
    const outPtr = ptr(outBuf);

    // Create zeros
    outBuf.fill(0);
    const zerosStatus = ffi.mlx_zeros(outPtr, ptr(shape), 2, DTYPE_TO_MLX.float32, s);
    expect(zerosStatus).toBe(0);
    const zerosCtx = unwrapPointer(readPtr(outPtr, 0), "mlx_zeros result");

    // Create ones
    outBuf.fill(0);
    const onesStatus = ffi.mlx_ones(outPtr, ptr(shape), 2, DTYPE_TO_MLX.float32, s);
    expect(onesStatus).toBe(0);
    const onesCtx = unwrapPointer(readPtr(outPtr, 0), "mlx_ones result");

    ffi.mlx_array_free(zerosCtx);
    ffi.mlx_array_free(onesCtx);
  });

  test("can perform matmul", () => {
    const s = defaultStream();

    const aData = new Float32Array([1, 0, 0, 1]);
    const aShape = new Int32Array([2, 2]);
    const a = ffi.mlx_array_new_data(ptr(aData), ptr(aShape), 2, DTYPE_TO_MLX.float32);

    const bData = new Float32Array([5, 6, 7, 8]);
    const b = ffi.mlx_array_new_data(ptr(bData), ptr(aShape), 2, DTYPE_TO_MLX.float32);

    const outBuf = new Uint8Array(8);
    const outPtr = ptr(outBuf);
    const status = ffi.mlx_matmul(outPtr, a, b, s);
    expect(status).toBe(0);

    const resultCtx = unwrapPointer(readPtr(outPtr, 0), "mlx_matmul result");
    ffi.mlx_array_eval(resultCtx);

    unwrapPointer(ffi.mlx_array_data_float32(resultCtx), "mlx_array_data_float32");

    ffi.mlx_array_free(a);
    ffi.mlx_array_free(b);
    ffi.mlx_array_free(resultCtx);
  });

  test("streams are valid", () => {
    const gpu = defaultStream();
    expect(gpu).not.toBeNull();
  });
});
