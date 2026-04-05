/**
 * Weight-name mapping for Gemma 3 text checkpoints.
 * @module
 */

import type { Gemma3TextConfig } from "./types";
import { isIgnoredGemma3TextWeight, sanitizeGemma3TextWeight } from "./types";

export function sanitizeGemma3Weight(
  config: Gemma3TextConfig,
  checkpointName: string,
): string | null {
  return sanitizeGemma3TextWeight(config, checkpointName);
}

export function isIgnoredGemma3Weight(config: Gemma3TextConfig, checkpointName: string): boolean {
  return isIgnoredGemma3TextWeight(config, checkpointName);
}
