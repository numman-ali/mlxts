/**
 * Weight-name mapping for dense Qwen3 text checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

/** Map Qwen3 checkpoint tensors onto the shared LLaMA-like parameter tree. */
export function sanitizeQwen3Weight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  return sanitizeLlamaLikeWeight(config, checkpointName);
}

/** Return true for Qwen3 checkpoint tensors intentionally absent from the runtime tree. */
export function isIgnoredQwen3Weight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return isIgnoredLlamaLikeWeight(config, checkpointName);
}
