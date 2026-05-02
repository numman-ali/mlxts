/**
 * Weight-name mapping for Gemma 3 text checkpoints.
 * @module
 */

import type { Gemma3TextConfig } from "./types";
import { isIgnoredGemma3TextWeight, sanitizeGemma3TextWeight } from "./types";

function stripLanguageModelPrefix(checkpointName: string): string {
  return checkpointName.startsWith("language_model.")
    ? checkpointName.slice("language_model.".length)
    : checkpointName;
}

export function sanitizeGemma3Weight(
  config: Gemma3TextConfig,
  checkpointName: string,
): string | null {
  return sanitizeGemma3TextWeight(config, stripLanguageModelPrefix(checkpointName));
}

export function isIgnoredGemma3Weight(config: Gemma3TextConfig, checkpointName: string): boolean {
  if (
    checkpointName.startsWith("vision_tower.") ||
    checkpointName.startsWith("multi_modal_projector.")
  ) {
    return true;
  }
  return isIgnoredGemma3TextWeight(config, stripLanguageModelPrefix(checkpointName));
}
