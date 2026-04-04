/**
 * MLX allocator and memory controls.
 *
 * These APIs expose the native allocator surface from mlx-c so long-running
 * training runs can observe memory pressure and set explicit limits.
 *
 * @module
 */

import { checkStatus } from "./error";
import { ffi } from "./ffi/lib";
import { ptr, sizeToNumber } from "./ffi/pointer";

/** Snapshot of the current MLX allocator state in bytes. */
export interface MemoryStats {
  activeBytes: number;
  cacheBytes: number;
  peakBytes: number;
  limitBytes: number;
}

function readSizeResult(label: string, callback: (out: BigUint64Array) => void): number {
  const out = new BigUint64Array(1);
  callback(out);
  const value = out[0];
  if (value === undefined) {
    throw new Error(`${label}: expected a size result`);
  }
  return sizeToNumber(value, label);
}

function setLimitWithPrevious(
  label: string,
  limitBytes: number,
  callback: (out: BigUint64Array, limit: bigint) => void,
): number {
  if (!Number.isSafeInteger(limitBytes) || limitBytes < 0) {
    throw new Error(`${label}: limitBytes must be a safe non-negative integer`);
  }

  const out = new BigUint64Array(1);
  callback(out, BigInt(limitBytes));
  const previous = out[0];
  if (previous === undefined) {
    throw new Error(`${label}: expected a previous limit result`);
  }
  return sizeToNumber(previous, label);
}

/** Current allocator usage in bytes. */
export function getActiveMemoryBytes(): number {
  return readSizeResult("mlx_get_active_memory", (out) => {
    checkStatus(ffi.mlx_get_active_memory(ptr(out)), "get_active_memory");
  });
}

/** Current allocator cache usage in bytes. */
export function getCacheMemoryBytes(): number {
  return readSizeResult("mlx_get_cache_memory", (out) => {
    checkStatus(ffi.mlx_get_cache_memory(ptr(out)), "get_cache_memory");
  });
}

/** Peak allocator usage in bytes since the last reset. */
export function getPeakMemoryBytes(): number {
  return readSizeResult("mlx_get_peak_memory", (out) => {
    checkStatus(ffi.mlx_get_peak_memory(ptr(out)), "get_peak_memory");
  });
}

/** Current allocator hard memory limit in bytes. */
export function getMemoryLimitBytes(): number {
  return readSizeResult("mlx_get_memory_limit", (out) => {
    checkStatus(ffi.mlx_get_memory_limit(ptr(out)), "get_memory_limit");
  });
}

/** Reset the allocator peak memory tracker. */
export function resetPeakMemory(): void {
  checkStatus(ffi.mlx_reset_peak_memory(), "reset_peak_memory");
}

/** Clear cached allocator memory that is no longer actively used. */
export function clearMemoryCache(): void {
  checkStatus(ffi.mlx_clear_cache(), "clear_cache");
}

/** Set the allocator cache limit in bytes and return the previous limit. */
export function setCacheLimitBytes(limitBytes: number): number {
  return setLimitWithPrevious("mlx_set_cache_limit", limitBytes, (out, limit) => {
    checkStatus(ffi.mlx_set_cache_limit(ptr(out), limit), "set_cache_limit");
  });
}

/** Set the allocator memory limit in bytes and return the previous limit. */
export function setMemoryLimitBytes(limitBytes: number): number {
  return setLimitWithPrevious("mlx_set_memory_limit", limitBytes, (out, limit) => {
    checkStatus(ffi.mlx_set_memory_limit(ptr(out), limit), "set_memory_limit");
  });
}

/** Set the allocator wired-memory limit in bytes and return the previous limit. */
export function setWiredLimitBytes(limitBytes: number): number {
  return setLimitWithPrevious("mlx_set_wired_limit", limitBytes, (out, limit) => {
    checkStatus(ffi.mlx_set_wired_limit(ptr(out), limit), "set_wired_limit");
  });
}

/** Read all currently exposed allocator stats in one call. */
export function getMemoryStats(): MemoryStats {
  return {
    activeBytes: getActiveMemoryBytes(),
    cacheBytes: getCacheMemoryBytes(),
    peakBytes: getPeakMemoryBytes(),
    limitBytes: getMemoryLimitBytes(),
  };
}
