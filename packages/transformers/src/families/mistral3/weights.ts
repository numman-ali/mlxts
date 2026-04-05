/**
 * Weight-name mapping for the text-decoder portion of Mistral 3 checkpoints.
 * @module
 */

import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredMistralWeight, sanitizeMistralWeight } from "../mistral/weights";

const LANGUAGE_MODEL_PREFIX = "language_model.";

export function sanitizeMistral3Weight(
  config: LlamaLikeConfig,
  checkpointName: string,
): string | null {
  if (!checkpointName.startsWith(LANGUAGE_MODEL_PREFIX)) {
    return null;
  }
  return sanitizeMistralWeight(config, checkpointName.slice(LANGUAGE_MODEL_PREFIX.length));
}

export function isIgnoredMistral3Weight(config: LlamaLikeConfig, checkpointName: string): boolean {
  if (!checkpointName.startsWith(LANGUAGE_MODEL_PREFIX)) {
    return true;
  }
  return isIgnoredMistralWeight(config, checkpointName.slice(LANGUAGE_MODEL_PREFIX.length));
}
