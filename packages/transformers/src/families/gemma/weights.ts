/**
 * Weight-name mapping for Gemma checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

export function sanitizeGemmaWeight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  return sanitizeLlamaLikeWeight(config, checkpointName);
}

export function isIgnoredGemmaWeight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return isIgnoredLlamaLikeWeight(config, checkpointName);
}
