/**
 * T5 encoder loading entry points.
 * @module
 */

import {
  inspectSnapshot,
  resolvePretrainedSnapshot,
  resolvePretrainedSource,
} from "../../pretrained/snapshot";
import type { LoadSourceOptions } from "../../pretrained/types";
import { parseT5EncoderConfig } from "./config";
import { T5EncoderModel } from "./model";
import {
  loadT5EncoderWeights,
  type T5EncoderWeightLoadOptions,
  type T5EncoderWeightLoadResult,
} from "./weights";

export type LoadT5EncoderModelOptions = LoadSourceOptions &
  T5EncoderWeightLoadOptions & {
    onWeightsLoaded?: (result: T5EncoderWeightLoadResult) => void;
  };

function emitWeightsProgress(
  options: LoadT5EncoderModelOptions,
  status: "weights-start" | "weights-complete",
  shardCount: number,
): void {
  options.onProgress?.({
    stage: "model",
    status,
    shardCount,
  });
}

/** Load a pretrained T5 encoder model from a local directory or Hugging Face repo. */
export async function loadT5EncoderModel(
  source: string,
  options: LoadT5EncoderModelOptions = {},
): Promise<T5EncoderModel> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const config = parseT5EncoderConfig(inspection.config);
  const model = new T5EncoderModel(config);
  const shardCount = inspection.model.safetensorPaths.length;
  emitWeightsProgress(options, "weights-start", shardCount);

  try {
    const result = await loadT5EncoderWeights(model, snapshot, options);
    options.onWeightsLoaded?.(result);
    emitWeightsProgress(options, "weights-complete", result.shardCount);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

export { resolvePretrainedSource as resolveT5EncoderModelSource };
