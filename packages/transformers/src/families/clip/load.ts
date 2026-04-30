/**
 * CLIP text encoder loading entry points.
 * @module
 */

import {
  inspectSnapshot,
  resolvePretrainedSnapshot,
  resolvePretrainedSource,
} from "../../pretrained/snapshot";
import type { LoadSourceOptions } from "../../pretrained/types";
import { parseCLIPTextConfig } from "./config";
import { CLIPTextModel, CLIPTextModelWithProjection } from "./model";
import {
  type CLIPTextWeightLoadOptions,
  type CLIPTextWeightLoadResult,
  loadCLIPTextWeights,
} from "./weights";

export type LoadCLIPTextModelOptions = LoadSourceOptions &
  CLIPTextWeightLoadOptions & {
    onWeightsLoaded?: (result: CLIPTextWeightLoadResult) => void;
  };

function emitWeightsProgress(
  options: LoadCLIPTextModelOptions,
  status: "weights-start" | "weights-complete",
  shardCount: number,
): void {
  options.onProgress?.({
    stage: "model",
    status,
    shardCount,
  });
}

async function loadCLIPTextModelInternal<
  TModel extends CLIPTextModel | CLIPTextModelWithProjection,
>(
  source: string,
  options: LoadCLIPTextModelOptions,
  createModel: (config: ReturnType<typeof parseCLIPTextConfig>) => TModel,
  target: "text" | "projected",
): Promise<TModel> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const config = parseCLIPTextConfig(inspection.config);
  const model = createModel(config);
  const shardCount = inspection.model.safetensorPaths.length;
  emitWeightsProgress(options, "weights-start", shardCount);

  try {
    const result = await loadCLIPTextWeights(model, snapshot, target, options);
    options.onWeightsLoaded?.(result);
    emitWeightsProgress(options, "weights-complete", result.shardCount);
    return model;
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

/** Load a pretrained CLIP text encoder from a local directory or Hugging Face repo. */
export async function loadCLIPTextModel(
  source: string,
  options: LoadCLIPTextModelOptions = {},
): Promise<CLIPTextModel> {
  return loadCLIPTextModelInternal(source, options, (config) => new CLIPTextModel(config), "text");
}

/** Load a pretrained projected CLIP text encoder from a local directory or Hugging Face repo. */
export async function loadCLIPTextModelWithProjection(
  source: string,
  options: LoadCLIPTextModelOptions = {},
): Promise<CLIPTextModelWithProjection> {
  return loadCLIPTextModelInternal(
    source,
    options,
    (config) => new CLIPTextModelWithProjection(config),
    "projected",
  );
}

export { resolvePretrainedSource as resolveCLIPTextModelSource };
