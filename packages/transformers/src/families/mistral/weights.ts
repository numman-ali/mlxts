/**
 * Weight-name mapping for Mistral checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

export function sanitizeMistralWeight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  return sanitizeLlamaLikeWeight(config, checkpointName);
}

export function isIgnoredMistralWeight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return isIgnoredLlamaLikeWeight(config, checkpointName);
}
