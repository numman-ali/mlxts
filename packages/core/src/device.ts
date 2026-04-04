/**
 * Device and stream management.
 *
 * MLX operations are dispatched to streams (CPU or GPU). Most users
 * never need to think about this — the default is GPU on Apple Silicon.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { checkStatus } from "./error";
import { ffi } from "./ffi/lib";
import { OutSlot, ptr, sizeToNumber, unwrapPointer } from "./ffi/pointer";

/** Device type for stream selection. */
export type DeviceType = "cpu" | "gpu";

// mlx-c device type enum values (from mlx/c/device.h)
const MLX_CPU = 0;
const MLX_GPU = 1;

// Lazy-initialized stream singletons.
// Streams are lightweight handles — we create them once and never free them.
let _gpuStream: Pointer | null = null;
let _cpuStream: Pointer | null = null;
let _defaultDevice: DeviceType | null = null;

function deviceTypeFromNative(raw: number): DeviceType {
  if (raw === MLX_CPU) {
    return "cpu";
  }
  if (raw === MLX_GPU) {
    return "gpu";
  }
  throw new Error(`Unknown mlx device type: ${raw}`);
}

function boolResult(label: string, callback: (out: Uint8Array) => void): boolean {
  const out = new Uint8Array(1);
  callback(out);
  const value = out[0];
  if (value === undefined) {
    throw new Error(`${label}: expected a boolean result`);
  }
  return value !== 0;
}

function i32Result(label: string, callback: (out: Int32Array) => void): number {
  const out = new Int32Array(1);
  callback(out);
  const value = out[0];
  if (value === undefined) {
    throw new Error(`${label}: expected a numeric result`);
  }
  return value;
}

/** Get the default GPU stream. */
export function gpuStream(): Pointer {
  if (_gpuStream === null) {
    _gpuStream = unwrapPointer(ffi.mlx_default_gpu_stream_new(), "mlx_default_gpu_stream_new");
  }
  return _gpuStream;
}

/** Get the default CPU stream. */
export function cpuStream(): Pointer {
  if (_cpuStream === null) {
    _cpuStream = unwrapPointer(ffi.mlx_default_cpu_stream_new(), "mlx_default_cpu_stream_new");
  }
  return _cpuStream;
}

/** Get the default stream (GPU unless changed via setDefaultDevice). */
export function defaultStream(): Pointer {
  return getDefaultDevice() === "gpu" ? gpuStream() : cpuStream();
}

/**
 * Set the default device for all operations.
 * This affects which stream is used when none is explicitly provided.
 */
export function setDefaultDevice(device: DeviceType): void {
  const mlxType = device === "gpu" ? MLX_GPU : MLX_CPU;
  const dev = unwrapPointer(ffi.mlx_device_new_type(mlxType, 0), "mlx_device_new_type");

  try {
    checkStatus(ffi.mlx_set_default_device(dev), "set_default_device");
    _defaultDevice = device;
  } finally {
    ffi.mlx_device_free(dev);
  }
}

/** Get the current default device type. */
export function getDefaultDevice(): DeviceType {
  if (_defaultDevice !== null) {
    return _defaultDevice;
  }
  const slot = new OutSlot();
  checkStatus(ffi.mlx_get_default_device(slot.prepare()), "get_default_device");
  const device = slot.read("default device");
  try {
    const rawType = i32Result("mlx_device_get_type", (out) => {
      checkStatus(ffi.mlx_device_get_type(ptr(out), device), "device_get_type");
    });
    const resolved = deviceTypeFromNative(rawType);
    _defaultDevice = resolved;
    return resolved;
  } finally {
    ffi.mlx_device_free(device);
  }
}

/** Wait for queued work on a stream to complete. */
export function synchronize(stream: Pointer = defaultStream()): void {
  checkStatus(ffi.mlx_synchronize(stream), "synchronize");
}

/** Number of available devices of a given type. */
export function deviceCount(device: DeviceType): number {
  const rawCount = i32Result("mlx_device_count", (out) => {
    checkStatus(
      ffi.mlx_device_count(ptr(out), device === "gpu" ? MLX_GPU : MLX_CPU),
      "device_count",
    );
  });
  return sizeToNumber(rawCount, "mlx_device_count");
}

/** Whether a specific device index is available. */
export function isDeviceAvailable(device: DeviceType, index = 0): boolean {
  const mlxType = device === "gpu" ? MLX_GPU : MLX_CPU;
  const dev = unwrapPointer(ffi.mlx_device_new_type(mlxType, index), "mlx_device_new_type");
  try {
    return boolResult("mlx_device_is_available", (out) => {
      checkStatus(ffi.mlx_device_is_available(ptr(out), dev), "device_is_available");
    });
  } finally {
    ffi.mlx_device_free(dev);
  }
}
