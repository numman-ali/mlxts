/**
 * Metal-specific profiling hooks.
 *
 * These are optional runtime helpers for Apple Silicon investigation work.
 *
 * @module
 */

import { checkStatus } from "./error";
import { ffi } from "./ffi/lib";
import { ptr } from "./ffi/pointer";

function boolResult(label: string, callback: (out: Uint8Array) => void): boolean {
  const out = new Uint8Array(1);
  callback(out);
  const value = out[0];
  if (value === undefined) {
    throw new Error(`${label}: expected a boolean result`);
  }
  return value !== 0;
}

/** Whether the current runtime has Metal support available. */
export function isMetalAvailable(): boolean {
  return boolResult("mlx_metal_is_available", (out) => {
    checkStatus(ffi.mlx_metal_is_available(ptr(out)), "metal_is_available");
  });
}

/** Start a Metal capture to the provided output path. */
export function startMetalCapture(path: string): void {
  if (path.length === 0) {
    throw new Error("startMetalCapture: path must not be empty");
  }
  const bytes = new TextEncoder().encode(`${path}\0`);
  checkStatus(ffi.mlx_metal_start_capture(ptr(bytes)), "metal_start_capture");
}

/** Stop the current Metal capture, if one is running. */
export function stopMetalCapture(): void {
  checkStatus(ffi.mlx_metal_stop_capture(), "metal_stop_capture");
}
