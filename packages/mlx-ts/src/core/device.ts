/**
 * Device and stream management.
 *
 * MLX operations are dispatched to streams (CPU or GPU). Most users
 * never need to think about this — the default is GPU on Apple Silicon.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { checkStatus } from "./error";
import { ffi, unwrapPointer } from "./ffi";

/** Device type for stream selection. */
export type DeviceType = "cpu" | "gpu";

// mlx-c device type enum values (from mlx/c/device.h)
const MLX_CPU = 0;
const MLX_GPU = 1;

// Lazy-initialized stream singletons.
// Streams are lightweight handles — we create them once and never free them.
let _gpuStream: Pointer | null = null;
let _cpuStream: Pointer | null = null;

/** The default device type used when no stream is explicitly passed. */
let _defaultDeviceType: DeviceType = "gpu";

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
  return _defaultDeviceType === "gpu" ? gpuStream() : cpuStream();
}

/**
 * Set the default device for all operations.
 * This affects which stream is used when none is explicitly provided.
 */
export function setDefaultDevice(device: DeviceType): void {
  _defaultDeviceType = device;

  const mlxType = device === "gpu" ? MLX_GPU : MLX_CPU;
  const dev = unwrapPointer(ffi.mlx_device_new_type(mlxType, 0), "mlx_device_new_type");

  try {
    checkStatus(ffi.mlx_set_default_device(dev), "set_default_device");
  } finally {
    ffi.mlx_device_free(dev);
  }
}

/** Get the current default device type. */
export function getDefaultDevice(): DeviceType {
  return _defaultDeviceType;
}
