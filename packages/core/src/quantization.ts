/**
 * Low-level MLX quantization primitives.
 *
 * These helpers intentionally expose the raw MLX building blocks without
 * introducing quantized modules or model-walking conveniences.
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, readResultArray, readResultPointer } from "./array";
import { defaultStream } from "./device";
import type { DType } from "./dtype";
import { checkStatus } from "./error";
import { ffi, OutSlot, optionalDType, optionalInt, ptr, sizeToNumber } from "./ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

export type QuantizationMode = "affine" | "mxfp4" | "mxfp8" | "nvfp4";

export type QuantizeOptions = {
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
  globalScale?: MxArray;
  stream?: S;
};

export type DequantizeOptions = {
  biases?: MxArray;
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
  globalScale?: MxArray;
  dtype?: DType;
  stream?: S;
};

export type QuantizedMatmulOptions = {
  biases?: MxArray;
  transpose?: boolean;
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
  stream?: S;
};

export type QuantizeResult = {
  weight: MxArray;
  scales: MxArray;
  biases?: MxArray;
};

const textEncoder = new TextEncoder();

function encodeMode(mode: QuantizationMode): Uint8Array {
  return textEncoder.encode(`${mode}\0`);
}

function readQuantizeOutputs(vector: Pointer): QuantizeResult {
  try {
    const count = sizeToNumber(ffi.mlx_vector_array_size(vector), "quantize_vec_size");
    if (count !== 2 && count !== 3) {
      throw new Error(`quantize: expected 2 or 3 outputs, received ${count}.`);
    }

    const slot = new OutSlot();
    const values: MxArray[] = [];
    for (let index = 0; index < count; index++) {
      checkStatus(ffi.mlx_vector_array_get(slot.prepare(), vector, index), `quantize_get_${index}`);
      values.push(MxArray._fromCtx(slot.read(`quantize result ${index}`)));
    }

    const weight = values[0];
    const scales = values[1];
    if (weight === undefined || scales === undefined) {
      throw new Error("quantize: missing required outputs.");
    }

    const biases = values[2];
    if (biases === undefined) {
      return { weight, scales };
    }

    return { weight, scales, biases };
  } finally {
    ffi.mlx_vector_array_free(vector);
  }
}

/** Quantize a weight tensor using MLX's native quantization kernels. */
export function quantize(weight: MxArray, options: QuantizeOptions = {}): QuantizeResult {
  const mode = options.mode ?? "affine";
  const encodedMode = encodeMode(mode);
  const outputs = readResultPointer("quantize outputs", (out) => {
    checkStatus(
      ffi.mlx_quantize(
        out,
        weight._ctx,
        optionalInt(options.groupSize),
        optionalInt(options.bits),
        ptr(encodedMode),
        options.globalScale?._ctx ?? null,
        s(options.stream),
      ),
      "quantize",
    );
  });

  return readQuantizeOutputs(outputs);
}

/** Dequantize a quantized tensor back into a dense MLX array. */
export function dequantize(
  weight: MxArray,
  scales: MxArray,
  options: DequantizeOptions = {},
): MxArray {
  const mode = options.mode ?? "affine";
  const encodedMode = encodeMode(mode);
  return readResultArray("dequantize", (out) => {
    checkStatus(
      ffi.mlx_dequantize(
        out,
        weight._ctx,
        scales._ctx,
        options.biases?._ctx ?? null,
        optionalInt(options.groupSize),
        optionalInt(options.bits),
        ptr(encodedMode),
        options.globalScale?._ctx ?? null,
        optionalDType(options.dtype),
        s(options.stream),
      ),
      "dequantize",
    );
  });
}

/** Multiply against packed quantized weights without materializing dense weights. */
export function quantizedMatmul(
  x: MxArray,
  weight: MxArray,
  scales: MxArray,
  options: QuantizedMatmulOptions = {},
): MxArray {
  const mode = options.mode ?? "affine";
  const encodedMode = encodeMode(mode);
  return readResultArray("quantizedMatmul", (out) => {
    checkStatus(
      ffi.mlx_quantized_matmul(
        out,
        x._ctx,
        weight._ctx,
        scales._ctx,
        options.biases?._ctx ?? null,
        options.transpose ?? false,
        optionalInt(options.groupSize),
        optionalInt(options.bits),
        ptr(encodedMode),
        s(options.stream),
      ),
      "quantizedMatmul",
    );
  });
}
