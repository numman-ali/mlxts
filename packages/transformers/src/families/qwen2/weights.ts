/**
 * Weight-name mapping for Qwen2 and Qwen2.5-VL text checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "../llama-like/types";

const LANGUAGE_MODEL_PREFIXES = ["model.language_model.", "language_model."] as const;
const VISION_PREFIXES = ["model.visual.", "visual."] as const;

function stripPrefix(checkpointName: string, prefixes: readonly string[]): string | null {
  for (const prefix of prefixes) {
    if (checkpointName.startsWith(prefix)) {
      return checkpointName.slice(prefix.length);
    }
  }
  return null;
}

function normalizeTextCheckpointName(checkpointName: string): string {
  if (
    checkpointName.startsWith("layers.") ||
    checkpointName.startsWith("embed_tokens.") ||
    checkpointName.startsWith("norm.")
  ) {
    return `model.${checkpointName}`;
  }
  return checkpointName;
}

/** Map Qwen2-family checkpoint tensors onto the shared LLaMA-like parameter tree. */
export function sanitizeQwen2Weight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  const languageModelName = stripPrefix(checkpointName, LANGUAGE_MODEL_PREFIXES);
  if (languageModelName !== null) {
    return sanitizeLlamaLikeWeight(config, normalizeTextCheckpointName(languageModelName));
  }

  if (stripPrefix(checkpointName, VISION_PREFIXES) !== null) {
    return null;
  }

  return sanitizeLlamaLikeWeight(config, checkpointName);
}

/** Return true for Qwen2-family tensors intentionally absent from the text runtime tree. */
export function isIgnoredQwen2Weight(config: LlamaLikeConfig, checkpointName: string): boolean {
  const languageModelName = stripPrefix(checkpointName, LANGUAGE_MODEL_PREFIXES);
  if (languageModelName !== null) {
    return isIgnoredLlamaLikeWeight(config, normalizeTextCheckpointName(languageModelName));
  }

  return stripPrefix(checkpointName, VISION_PREFIXES) !== null
    ? true
    : isIgnoredLlamaLikeWeight(config, checkpointName);
}
