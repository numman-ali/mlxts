/**
 * Weight-name mapping for Phi-3 checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

export function sanitizePhiWeight(config: LlamaLikeConfig, checkpointName: string): string | null {
  return sanitizeLlamaLikeWeight(config, checkpointName);
}

export function isIgnoredPhiWeight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return isIgnoredLlamaLikeWeight(config, checkpointName);
}
