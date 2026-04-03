/**
 * Pointer utilities for Bun FFI.
 *
 * Bridges Bun's branded Pointer type with the rest of the codebase.
 * All pointer narrowing and conversion happens here — higher-level code
 * works with Pointer directly but never needs to assert or cast.
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import {
  ptr as bunPtr,
  read as bunRead,
  toArrayBuffer as bunToArrayBuffer,
  CString,
} from "bun:ffi";

export type { Pointer };

type NativeBufferView = NodeJS.TypedArray | ArrayBufferLike | DataView;

/** Get a native pointer from a TypedArray. */
export function ptr(view: NativeBufferView, byteOffset?: number): Pointer {
  return bunPtr(view, byteOffset);
}

/** Read a pointer-sized value from native memory. */
export function readPtr(address: Pointer, byteOffset: number): Pointer | null {
  const rawAddress = bunRead.ptr(address, byteOffset);
  return isPointer(rawAddress) ? rawAddress : null;
}

/** Read an i32 from native memory. */
export function readI32(address: Pointer, byteOffset: number): number {
  return bunRead.i32(address, byteOffset);
}

/** Create an ArrayBuffer viewing native memory. */
export function nativeSlice(address: Pointer, byteOffset: number, byteLength: number): ArrayBuffer {
  return bunToArrayBuffer(address, byteOffset, byteLength);
}

/** Convert a C string pointer to a JS string. */
export function readCString(pointer: Pointer): string {
  return new CString(pointer).toString();
}

/** Require a non-null native pointer from an FFI call. */
export function unwrapPointer(pointer: Pointer | null, source: string): Pointer {
  if (pointer === null) {
    throw new Error(`${source} returned a null pointer`);
  }
  return pointer;
}

/** Convert a size_t-like return value to a safe JavaScript number. */
export function sizeToNumber(value: number | bigint, source: string): number {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${source} returned ${value}, which is not a safe non-negative integer`);
    }
    return value;
  }

  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`${source} returned ${value}, which exceeds JS safe integer range`);
  }

  return Number(value);
}

function isPointer(address: number): address is Pointer {
  return address !== 0;
}

/**
 * A reusable 8-byte output slot for FFI calls that write results via pointer.
 *
 * mlx-c operations write their results into output pointer parameters
 * (e.g., `mlx_array*`, `mlx_vector_array*`). Each slot holds one such
 * pointer-sized struct (`{ void* ctx }`).
 *
 * Use `prepare()` before each FFI call to zero the buffer, then `read()`
 * after to extract the result pointer.
 *
 * Unlike the global `prepareOut`/`readOut` in array.ts, OutSlot instances
 * can be used concurrently (e.g., for value + gradient output buffers).
 */
export class OutSlot {
  private readonly buf = new Uint8Array(8);
  readonly ptr: Pointer;

  constructor() {
    this.ptr = ptr(this.buf);
  }

  /** Zero the buffer and return its address for use as an output pointer. */
  prepare(): Pointer {
    this.buf.fill(0);
    return this.ptr;
  }

  /** Read the result pointer from the buffer after an FFI call. */
  read(label: string): Pointer {
    return unwrapPointer(readPtr(this.ptr, 0), label);
  }
}
