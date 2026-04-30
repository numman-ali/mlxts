/**
 * Padding operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { array, type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi, ptr } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

const PAD_MODE_BYTES = {
  constant: new TextEncoder().encode("constant\0"),
} as const;

export type PadMode = keyof typeof PAD_MODE_BYTES;

/** Pad an array with a constant value along each axis. */
export function pad(
  a: MxArray,
  padWidths: readonly (readonly [number, number])[],
  value = 0,
  mode: PadMode = "constant",
  stream?: S,
): MxArray {
  if (padWidths.length !== a.shape.length) {
    throw new Error(
      `pad: expected ${a.shape.length} pad-width pairs for shape ${a.shape.join("x")}.`,
    );
  }
  const lowPadValues: number[] = [];
  const highPadValues: number[] = [];
  const outputShape = a.shape.map((dimension, axis) => {
    const pair = padWidths[axis];
    const lowPad = pair?.[0];
    const highPad = pair?.[1];
    if (
      pair === undefined ||
      pair.length !== 2 ||
      lowPad === undefined ||
      highPad === undefined ||
      !Number.isInteger(lowPad) ||
      !Number.isInteger(highPad) ||
      lowPad < 0 ||
      highPad < 0
    ) {
      throw new Error(`pad: padWidths[${axis}] must be a non-negative integer pair.`);
    }
    lowPadValues.push(lowPad);
    highPadValues.push(highPad);
    return dimension + lowPad + highPad;
  });
  const axes = Int32Array.from(a.shape.map((_, axis) => axis));
  const lowPads = Int32Array.from(lowPadValues);
  const highPads = Int32Array.from(highPadValues);
  const modeBytes = PAD_MODE_BYTES[mode];
  using padValue = array(value, a.dtype);
  return readResultArrayWithMetadata("pad", { shape: outputShape, dtype: a.dtype }, (out) => {
    checkStatus(
      ffi.mlx_pad(
        out,
        a._ctx,
        ptr(axes),
        axes.length,
        ptr(lowPads),
        lowPads.length,
        ptr(highPads),
        highPads.length,
        padValue._ctx,
        ptr(modeBytes),
        s(stream),
      ),
      "pad",
    );
  });
}
