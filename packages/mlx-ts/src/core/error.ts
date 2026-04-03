/**
 * Error handling for mlx-c FFI calls.
 *
 * Registers a global error handler with mlx-c to capture error messages,
 * and provides checkStatus() to throw typed errors after every FFI call.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { FFIType, JSCallback } from "bun:ffi";

import { ffi, readCString, unwrapPointer } from "./ffi";

/** Typed error for mlx-c operation failures. */
export class MxError extends Error {
  override readonly name = "MxError";
  readonly operation: string;
  readonly code: number;

  constructor(operation: string, code: number, mlxMessage: string) {
    const msg = mlxMessage
      ? `${operation} failed: ${mlxMessage}`
      : `${operation} failed with code ${code}`;
    super(msg);
    this.operation = operation;
    this.code = code;
  }
}

// Module-level storage for the last error message from mlx-c
let lastErrorMessage = "";

// The JSCallback must be kept alive for the lifetime of the library.
let errorHandlerCallback: JSCallback | null = null;
let errorHandlerDestructor: JSCallback | null = null;

/**
 * Register an error handler with mlx-c.
 * Called once at library initialization.
 */
export function initializeErrorHandler(): void {
  if (errorHandlerCallback) return;

  errorHandlerCallback = new JSCallback(
    (msgPtr: Pointer | null, _data: Pointer | null) => {
      if (msgPtr !== null) {
        lastErrorMessage = readCString(msgPtr);
      }
    },
    { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.void },
  );

  errorHandlerDestructor = new JSCallback((_data: Pointer | null) => {}, {
    args: [FFIType.ptr],
    returns: FFIType.void,
  });

  ffi.mlx_set_error_handler(
    unwrapPointer(errorHandlerCallback.ptr, "error handler callback"),
    null,
    unwrapPointer(errorHandlerDestructor.ptr, "error handler destructor"),
  );
}

/**
 * Check the return code of an mlx-c call. Throws MxError on failure.
 */
export function checkStatus(code: number, operation: string): void {
  if (code !== 0) {
    const msg = lastErrorMessage;
    lastErrorMessage = "";
    throw new MxError(operation, code, msg);
  }
}
