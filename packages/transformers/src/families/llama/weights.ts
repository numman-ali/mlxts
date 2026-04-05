/**
 * Weight-name mapping for LLaMA checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

export function sanitizeLlamaWeight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  return sanitizeLlamaLikeWeight(config, checkpointName);
}

export function isIgnoredLlamaWeight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return isIgnoredLlamaLikeWeight(config, checkpointName);
}
