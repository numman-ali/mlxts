/**
 * Whisper loading entry points.
 * @module
 */

import {
  inspectSnapshot,
  resolvePretrainedSnapshot,
  resolvePretrainedSource,
} from "../../pretrained/snapshot";
import type { LoadSourceOptions } from "../../pretrained/types";
import { parseWhisperConfig } from "./config";
import { WhisperForConditionalGeneration } from "./model";
import {
  loadWhisperWeights,
  type WhisperWeightLoadOptions,
  type WhisperWeightLoadResult,
} from "./weights";

export type LoadWhisperModelOptions = LoadSourceOptions &
  WhisperWeightLoadOptions & {
    onWeightsLoaded?: (result: WhisperWeightLoadResult) => void;
  };

function emitWeightsProgress(
  options: LoadWhisperModelOptions,
  status: "weights-start" | "weights-complete",
  shardCount: number,
): void {
  options.onProgress?.({
    stage: "model",
    status,
    shardCount,
  });
}

/** Load a pretrained Whisper conditional-generation model. */
export async function loadWhisperModel(
  source: string,
  options: LoadWhisperModelOptions = {},
): Promise<WhisperForConditionalGeneration> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const config = parseWhisperConfig(inspection.config);
  const model = new WhisperForConditionalGeneration(config);
  const shardCount = inspection.model.safetensorPaths.length;
  emitWeightsProgress(options, "weights-start", shardCount);

  try {
    const result = await loadWhisperWeights(model, snapshot, options);
    options.onWeightsLoaded?.(result);
    emitWeightsProgress(options, "weights-complete", result.shardCount);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

export { resolvePretrainedSource as resolveWhisperModelSource };
