/**
 * FFI boundary package — the sole point of contact with the native library.
 *
 * Re-exports everything that consumer code needs: the `ffi` symbols object,
 * pointer utilities, and type re-exports. All native pointer narrowing,
 * type assertions, and C-level details stay within this package.
 *
 * @module
 */

// --- Closure bridge (internal, not re-exported from public index.ts) ---
export { extractAllArrays, extractSingleArray, type GradFn, withClosure } from "./closure-bridge";
// --- Native library symbols ---
export { ffi } from "./lib";

// --- Pointer utilities ---
export {
  nativeSlice,
  OutSlot,
  type Pointer,
  ptr,
  readCString,
  readI32,
  readPtr,
  sizeToNumber,
  unwrapPointer,
} from "./pointer";
