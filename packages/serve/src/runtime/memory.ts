/**
 * Best-effort MLX allocator telemetry for serving logs.
 * @module
 */

import { getMemoryStats } from "@mlxts/core";
import type { GenerationMemoryUsage } from "../types";

/** Read current MLX memory stats without letting telemetry break serving. */
export function readGenerationMemoryUsage(): GenerationMemoryUsage | undefined {
  try {
    const stats = getMemoryStats();
    return {
      activeBytes: stats.activeBytes,
      cacheBytes: stats.cacheBytes,
      peakBytes: stats.peakBytes,
      limitBytes: stats.limitBytes,
    };
  } catch {
    return undefined;
  }
}
